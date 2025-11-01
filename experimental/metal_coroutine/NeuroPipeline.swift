//  NeuroPipeline.swift
//  MetalCoroutinesTest
//
//  Created by Sydney Bach on 2/24/25.

import Cocoa
import Metal
import Foundation
import Accelerate
import MetalPerformanceShaders

actor NeuroPipeline {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelineState: MTLComputePipelineState

    private var modelParameters: ModelParameters
    private let logger: MetalLogger
    private var stepNum: UInt32 = 0
    nonisolated private let diagnostics = DiagnosticLogger.shared

    private var h_liquid_read_buffer: MTLBuffer!
    private var h_liquid_write_buffer: MTLBuffer!
    private var c_t_buffer: MTLBuffer!
    private var n_t_buffer: MTLBuffer!
    private var gate_mask_buffer: MTLBuffer!
    private var lambda_mask_buffer: MTLBuffer!
    private var W_recurrent_buffer: MTLBuffer!
    private var W_i_buffer: MTLBuffer!
    private var U_i_buffer: MTLBuffer!
    private var b_i_buffer: MTLBuffer!
    private var W_f_buffer: MTLBuffer!
    private var U_f_buffer: MTLBuffer!
    private var b_f_buffer: MTLBuffer!
    private var W_o_buffer: MTLBuffer!
    private var U_o_buffer: MTLBuffer!
    private var b_o_buffer: MTLBuffer!
    private var W_g_buffer: MTLBuffer!
    private var U_g_buffer: MTLBuffer!
    private var b_g_buffer: MTLBuffer!
    private var lambda_buffer: MTLBuffer!
    private var logBuffer: MTLBuffer!
    private var lastUpdateTime: CFAbsoluteTime

    init(parameters: ModelParameters, logger: MetalLogger) async throws {
        self.modelParameters = parameters
        self.logger = logger
        self.lastUpdateTime = CFAbsoluteTimeGetCurrent()
        guard let device = MTLCreateSystemDefaultDevice() else { throw MetalError.deviceCreationFailed }
        self.device = device
        guard let commandQueue = device.makeCommandQueue() else { throw MetalError.commandBufferCreationFailed }
        self.commandQueue = commandQueue
        self.pipelineState = try Self.makePipelineState(device: device, logger: self.logger)
        try await self.allocateBuffers()
    }

    static func makePipelineState(device: MTLDevice, logger: MetalLogger) throws -> MTLComputePipelineState {
        if let library = device.makeDefaultLibrary(), let kernelFunction = library.makeFunction(name: "liquid_cfc_xlstm_kernel") {
            return try device.makeComputePipelineState(function: kernelFunction)
        }
        throw MetalError.kernelFunctionNotFound
    }

    private func allocateBuffers() async throws {
        func buf(_ count: Int) -> MTLBuffer? { device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared) }
        h_liquid_read_buffer = buf(modelParameters.N)
        h_liquid_write_buffer = buf(modelParameters.N)
        c_t_buffer = buf(modelParameters.N)
        n_t_buffer = buf(modelParameters.N)
        let recurrentSize = modelParameters.N * modelParameters.N * MemoryLayout<Float>.size
        W_recurrent_buffer = device.makeBuffer(bytes: modelParameters.W_recurrent, length: recurrentSize, options: .storageModeShared)
        // Other parameter buffers omitted for brevity in this experimental file
        logBuffer = device.makeBuffer(length: 1024, options: .storageModeShared)
    }

    func executeStep() async throws -> [Float] {
        var params = MetalKernelParams(
            N: Int32(modelParameters.N),
            dt: modelParameters.dt,
            num_steps: 1,
            alpha: modelParameters.alpha,
            target_sum: modelParameters.target_sum,
            neural_clock: 0.5,
            step_num: stepNum,
            eta: modelParameters.eta,
            use_hebbian: modelParameters.use_hebbian,
            decay_rate: modelParameters.decay_rate
        )
        guard let paramsBuffer = device.makeBuffer(bytes: &params, length: MemoryLayout<MetalKernelParams>.size, options: .storageModeShared) else {
            throw MetalError.bufferCreationFailed
        }
        guard let commandBuffer = commandQueue.makeCommandBuffer(), let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandBufferCreationFailed
        }
        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 0)
        encoder.setBuffer(W_recurrent_buffer, offset: 0, index: 1)
        encoder.setBuffer(h_liquid_read_buffer, offset: 0, index: 17)
        encoder.setBuffer(h_liquid_write_buffer, offset: 0, index: 18)
        encoder.setBuffer(c_t_buffer, offset: 0, index: 19)
        encoder.setBuffer(n_t_buffer, offset: 0, index: 20)
        encoder.setBuffer(logBuffer, offset: 0, index: 21)
        let TILE_SIZE = 16
        let tg = MTLSize(width: TILE_SIZE, height: TILE_SIZE, depth: 1)
        let ng = MTLSize(width: (modelParameters.N + TILE_SIZE - 1) / TILE_SIZE, height: (modelParameters.N + TILE_SIZE - 1) / TILE_SIZE, depth: 1)
        encoder.dispatchThreadgroups(ng, threadsPerThreadgroup: tg)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        stepNum += 1
        var result = [Float](repeating: 0, count: modelParameters.N)
        if let buf = (stepNum % 2 == 0) ? h_liquid_read_buffer : h_liquid_write_buffer {
            memcpy(&result, buf.contents(), result.count * MemoryLayout<Float>.size)
        }
        return result
    }
}

// Minimal supporting types (stubs for experiment)
struct MetalKernelParams { var N:Int32; var dt:Float; var num_steps:Int32; var alpha:Float; var target_sum:Float; var neural_clock:Float; var step_num:UInt32; var eta:Float; var use_hebbian:Bool; var decay_rate:Float }
struct ModelParameters { var N:Int; var dt:Float; var alpha:Float; var target_sum:Float; var eta:Float; var decay_rate:Float; var use_hebbian:Bool; var W_recurrent:[Float] }
enum MetalError: Error { case deviceCreationFailed, commandBufferCreationFailed, bufferCreationFailed, kernelFunctionNotFound }
class MetalLogger { func log(_ s:String){} }
class DiagnosticLogger { static let shared = DiagnosticLogger(); func log(_ s:String){} }

