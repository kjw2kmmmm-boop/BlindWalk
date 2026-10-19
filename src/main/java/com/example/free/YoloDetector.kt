package com.example.free

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.util.Collections
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

class YoloDetector(private val context: Context) {

    companion object {
        const val INPUT_SIZE = 320
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()

    private var brailleSession: OrtSession? = null
    private var obstacleSession: OrtSession? = null

    // 두 모델을 동시에 병렬로 돌리기 위한 전용 스레드 풀
    private val parallelExecutor = Executors.newFixedThreadPool(2)

    // 실제 추론에 걸린 시간(ms) 기록용
    var lastInferenceTimeMs: Long = 0L
        private set

    private val brailleClasses = arrayOf("파손블록", "점자블록")

    private val obstacleClassMap = mapOf(
        0 to "사람",
        1 to "자전거",
        2 to "자동차",
        3 to "오토바이",
        5 to "버스",
        7 to "트럭",
        9 to "신호등",
        10 to "소화전",
        11 to "정지표지판",
        13 to "벤치",
        15 to "고양이",
        16 to "개",
        24 to "가방/짐",
        28 to "캐리어",
        56 to "의자",
        58 to "화분",
        60 to "탁자"
    )

    private val imgDataBuffer: FloatBuffer = FloatBuffer.allocate(1 * 3 * INPUT_SIZE * INPUT_SIZE)
    private val intValuesBuffer = IntArray(INPUT_SIZE * INPUT_SIZE)
    private val inputShape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())

    init {
        // 병렬 실행 시 코어 충돌을 막기 위해 각 세션당 2코어씩 배정 (총 4코어 풀가동)
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            try {
                addXnnpack(emptyMap())
            } catch (e: Exception) {
                // ignore
            }
        }

        val brailleBytes = context.assets.open("best.onnx").readBytes()
        brailleSession = env.createSession(brailleBytes, options)

        // [핵심] 무거운 yolo26n-seg.onnx 대신 가벼운 박스 전용 yolo26n.onnx 로드!
        val obstacleBytes = context.assets.open("yolo26n.onnx").readBytes()
        obstacleSession = env.createSession(obstacleBytes, options)
    }

    data class DetectionResult(
        val category: String,
        val classId: Int,
        val className: String,
        val score: Float,
        val distanceM: Float,
        val screenPos: String,
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float
    )

    fun detectFast(bitmap320: Bitmap): List<DetectionResult> {
        val bSession = brailleSession ?: return emptyList()
        val oSession = obstacleSession ?: return emptyList()

        val startTime = System.currentTimeMillis()

        preprocessBitmapInPlace(bitmap320)
        val inputTensor = OnnxTensor.createTensor(env, imgDataBuffer, inputShape)

        val combinedResults = mutableListOf<DetectionResult>()

        try {
            // [핵심] 두 모델을 기다리지 않고 2개의 CPU 스레드에서 '동시에' 병렬 실행!
            val obstacleFuture = parallelExecutor.submit<List<DetectionResult>> {
                val oInputName = oSession.inputNames.iterator().next()
                oSession.run(Collections.singletonMap(oInputName, inputTensor)).use { oOut ->
                    val tensor = oOut.get(0) as OnnxTensor
                    parseTensorFast(tensor, 80, 0.45f, false)
                }
            }

            val brailleFuture = parallelExecutor.submit<List<DetectionResult>> {
                val bInputName = bSession.inputNames.iterator().next()
                bSession.run(Collections.singletonMap(bInputName, inputTensor)).use { bOut ->
                    val tensor = bOut.get(0) as OnnxTensor
                    parseTensorFast(tensor, 2, 0.65f, true)
                }
            }

            // 두 모델의 결과 합치기
            combinedResults.addAll(obstacleFuture.get())
            combinedResults.addAll(brailleFuture.get())

        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            inputTensor.close()
        }

        lastInferenceTimeMs = System.currentTimeMillis() - startTime
        return combinedResults
    }

    private fun preprocessBitmapInPlace(bitmap: Bitmap) {
        bitmap.getPixels(intValuesBuffer, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        val area = INPUT_SIZE * INPUT_SIZE
        val inv255 = 1.0f / 255.0f

        for (i in 0 until area) {
            val pixel = intValuesBuffer[i]
            imgDataBuffer.put(i, ((pixel shr 16) and 0xFF) * inv255)
            imgDataBuffer.put(i + area, ((pixel shr 8) and 0xFF) * inv255)
            imgDataBuffer.put(i + 2 * area, (pixel and 0xFF) * inv255)
        }
        imgDataBuffer.rewind()
    }

    private fun parseTensorFast(
        tensor: OnnxTensor,
        numClasses: Int,
        confThreshold: Float,
        isBrailleModel: Boolean
    ): List<DetectionResult> {
        val candidates = mutableListOf<DetectionResult>()
        val shape = tensor.info.shape
        val fb = tensor.floatBuffer
        fb.rewind()

        // YOLO26 기본 Detect 모델이 End-to-End NMS 내장 형태([1, 300, 6])로 출력될 경우 대응
        if (shape.size == 3 && shape[2].toInt() == 6) {
            val numDet = shape[1].toInt()
            for (i in 0 until numDet) {
                val base = i * 6
                val score = fb.get(base + 4)
                if (score > confThreshold) {
                    val x1 = fb.get(base + 0)
                    val y1 = fb.get(base + 1)
                    val x2 = fb.get(base + 2)
                    val y2 = fb.get(base + 3)
                    val clsId = fb.get(base + 5).toInt()
                    val cx = (x1 + x2) / 2f
                    val cy = (y1 + y2) / 2f
                    val w = x2 - x1
                    val h = y2 - y1
                    createResultIfValid(clsId, score, cx, cy, w, h, isBrailleModel)?.let {
                        candidates.add(it)
                    }
                }
            }
            return candidates.take(5)
        }

        val dim1 = shape[1].toInt()
        val dim2 = shape[2].toInt()

        if (dim1 < dim2) {
            val numChannels = dim1
            val numAnchors = dim2
            val classEnd = minOf(4 + numClasses, numChannels)

            for (i in 0 until numAnchors) {
                var maxScore = 0f
                var bestClassId = -1

                for (c in 4 until classEnd) {
                    val score = fb.get(c * numAnchors + i)
                    if (score > maxScore) {
                        maxScore = score
                        bestClassId = c - 4
                    }
                }

                if (maxScore > confThreshold) {
                    val cx = fb.get(0 * numAnchors + i)
                    val cy = fb.get(1 * numAnchors + i)
                    val w = fb.get(2 * numAnchors + i)
                    val h = fb.get(3 * numAnchors + i)

                    createResultIfValid(bestClassId, maxScore, cx, cy, w, h, isBrailleModel)?.let {
                        candidates.add(it)
                    }
                }
            }
        } else {
            val numAnchors = dim1
            val numChannels = dim2
            val classEnd = minOf(4 + numClasses, numChannels)

            for (i in 0 until numAnchors) {
                val baseIdx = i * numChannels
                var maxScore = 0f
                var bestClassId = -1

                for (c in 4 until classEnd) {
                    val score = fb.get(baseIdx + c)
                    if (score > maxScore) {
                        maxScore = score
                        bestClassId = c - 4
                    }
                }

                if (maxScore > confThreshold) {
                    val cx = fb.get(baseIdx + 0)
                    val cy = fb.get(baseIdx + 1)
                    val w = fb.get(baseIdx + 2)
                    val h = fb.get(baseIdx + 3)

                    createResultIfValid(bestClassId, maxScore, cx, cy, w, h, isBrailleModel)?.let {
                        candidates.add(it)
                    }
                }
            }
        }

        return applyNMS(candidates, iouThreshold = 0.45f).take(5)
    }

    private fun createResultIfValid(
        classId: Int,
        score: Float,
        cx: Float,
        cy: Float,
        w: Float,
        h: Float,
        isBrailleModel: Boolean
    ): DetectionResult? {
        val className: String
        val category: String

        if (isBrailleModel) {
            if (classId !in brailleClasses.indices) return null
            className = brailleClasses[classId]
            category = if (classId == 0) "BROKEN" else "BRAILLE"
        } else {
            className = obstacleClassMap[classId] ?: return null
            category = "OBSTACLE"
        }

        val maxCoord = INPUT_SIZE.toFloat()
        val x1 = max(0f, cx - w / 2f)
        val y1 = max(0f, cy - h / 2f)
        val x2 = min(maxCoord, cx + w / 2f)
        val y2 = min(maxCoord, cy + h / 2f)

        val boxHeight = max(1f, y2 - y1)
        val distanceM = (225f / boxHeight).coerceIn(0.5f, 10.0f)

        val screenPos = when {
            cx < maxCoord / 3f -> "왼쪽"
            cx > maxCoord * 2f / 3f -> "오른쪽"
            else -> "정면"
        }

        return DetectionResult(
            category = category,
            classId = classId,
            className = className,
            score = score,
            distanceM = distanceM,
            screenPos = screenPos,
            x1 = x1,
            y1 = y1,
            x2 = x2,
            y2 = y2
        )
    }

    private fun applyNMS(
        boxes: List<DetectionResult>,
        iouThreshold: Float
    ): List<DetectionResult> {
        val sorted = boxes.sortedByDescending { it.score }.toMutableList()
        val selected = mutableListOf<DetectionResult>()

        while (sorted.isNotEmpty()) {
            val current = sorted.removeAt(0)
            selected.add(current)

            val iterator = sorted.iterator()
            while (iterator.hasNext()) {
                val next = iterator.next()
                if (current.className == next.className && calculateIoU(current, next) > iouThreshold) {
                    iterator.remove()
                }
            }
        }
        return selected
    }

    private fun calculateIoU(a: DetectionResult, b: DetectionResult): Float {
        val interLeft = max(a.x1, b.x1)
        val interTop = max(a.y1, b.y1)
        val interRight = min(a.x2, b.x2)
        val interBottom = min(a.y2, b.y2)

        val interWidth = max(0f, interRight - interLeft)
        val interHeight = max(0f, interBottom - interTop)
        val interArea = interWidth * interHeight

        val areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
        val areaB = (b.x2 - b.x1) * (b.y2 - b.y1)

        return interArea / (areaA + areaB - interArea + 1e-5f)
    }

    fun close() {
        parallelExecutor.shutdown()
        brailleSession?.close()
        obstacleSession?.close()
        env.close()
    }
}