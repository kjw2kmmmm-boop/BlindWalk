package com.example.free

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var boundingBoxView: BoundingBoxView
    private lateinit var yoloDetector: YoloDetector

    private val analysisExecutor = Executors.newSingleThreadExecutor()

    private val cameraPermissionLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted ->
            if (granted) {
                startCamera()
            } else {
                finish()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        previewView.scaleType = PreviewView.ScaleType.FIT_CENTER

        boundingBoxView = findViewById(R.id.boundingBoxView)
        yoloDetector = YoloDetector(this)

        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(
                Manifest.permission.CAMERA
            )
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build()
                .also {
                    it.surfaceProvider = previewView.surfaceProvider
                }

            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(analysisExecutor) { imageProxy ->
                        try {
                            val rawBitmap = imageProxy.toBitmap()
                            val rotationDegrees = imageProxy.imageInfo.rotationDegrees

                            // 회전 후 실제 화면에 표시될 가로/세로 비율 계산
                            val rotatedWidth = if (rotationDegrees % 180 != 0) rawBitmap.height else rawBitmap.width
                            val rotatedHeight = if (rotationDegrees % 180 != 0) rawBitmap.width else rawBitmap.height

                            // [초고속 최적화] 고해상도 회전 + 640x640 축소를 한 번에 처리
                            val matrix = Matrix().apply {
                                postRotate(rotationDegrees.toFloat())
                                val scaleX = 320f / rotatedWidth.toFloat()
                                val scaleY = 320f / rotatedHeight.toFloat()
                                postScale(scaleX, scaleY)
                            }

                            val bitmap320 = Bitmap.createBitmap(
                                rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, false
                            )

                            val results = yoloDetector.detectFast(bitmap320)

                            runOnUiThread {
                                boundingBoxView.setResults(
                                    results,
                                    rotatedWidth,
                                    rotatedHeight,
                                    yoloDetector.lastInferenceTimeMs // 실제 추론 속도 전달
                                )
                            }

                        } catch (e: Exception) {
                            e.printStackTrace()
                        } finally {
                            imageProxy.close()
                        }
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    imageAnalysis
                )
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
        yoloDetector.close()
    }
}