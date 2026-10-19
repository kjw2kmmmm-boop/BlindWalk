package com.example.free

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

class BoundingBoxView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private var results: List<YoloDetector.DetectionResult> = emptyList()
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1
    private var inferenceTimeMs: Long = 0L

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 8f
        isAntiAlias = true
    }

    private val textBgPaint = Paint().apply {
        color = Color.argb(180, 0, 0, 0)
        style = Paint.Style.FILL
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 38f
        style = Paint.Style.FILL
        isFakeBoldText = true
        isAntiAlias = true
    }

    private val fpsPaint = Paint().apply {
        color = Color.CYAN
        textSize = 42f
        style = Paint.Style.FILL
        isFakeBoldText = true
        isAntiAlias = true
    }

    fun setResults(
        newResults: List<YoloDetector.DetectionResult>,
        srcWidth: Int,
        srcHeight: Int,
        timeMs: Long = 0L
    ) {
        results = newResults
        imageWidth = srcWidth
        imageHeight = srcHeight
        inferenceTimeMs = timeMs
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (imageWidth <= 0 || imageHeight <= 0) return

        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()

        val scale = min(viewWidth / imageWidth.toFloat(), viewHeight / imageHeight.toFloat())
        val displayedWidth = imageWidth * scale
        val displayedHeight = imageHeight * scale

        val offsetX = (viewWidth - displayedWidth) / 2f
        val offsetY = (viewHeight - displayedHeight) / 2f

        val scaleX = displayedWidth / 320f
        val scaleY = displayedHeight / 320f

        // 좌측 상단에 실제 AI 추론 시간(ms) 및 초당 추론 횟수(FPS) 표시
        if (inferenceTimeMs > 0) {
            val fps = 1000f / inferenceTimeMs.toFloat()
            val perfText = "인식 속도: ${inferenceTimeMs}ms (${String.format("%.1f", fps)}회/초)"
            canvas.drawRect(20f, 80f, 620f, 145f, textBgPaint)
            canvas.drawText(perfText, 35f, 128f, fpsPaint)
        }

        for (result in results) {
            val left = offsetX + result.x1 * scaleX
            val top = offsetY + result.y1 * scaleY
            val right = offsetX + result.x2 * scaleX
            val bottom = offsetY + result.y2 * scaleY

            boxPaint.color = when (result.category) {
                "BRAILLE" -> Color.GREEN
                "BROKEN" -> Color.RED
                else -> Color.rgb(255, 165, 0)
            }

            canvas.drawRect(left, top, right, bottom, boxPaint)

            val label = "[${result.className}] ${result.screenPos} ${String.format("%.1f", result.distanceM)}m (${String.format("%.2f", result.score)})"
            val textY = maxOf(offsetY + 45f, top - 12f)
            val textWidth = textPaint.measureText(label)

            canvas.drawRect(left, textY - 42f, left + textWidth + 20f, textY + 10f, textBgPaint)
            canvas.drawText(label, left + 10f, textY, textPaint)
        }
    }
}