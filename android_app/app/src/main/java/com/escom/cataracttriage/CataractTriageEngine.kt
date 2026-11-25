package com.escom.cataracttriage

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.json.JSONObject
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max

/**
 * Offline ONNX Runtime Mobile inference engine + zero-backprop Grad-CAM overlay.
 */
class CataractTriageEngine(context: Context) : AutoCloseable {

    data class TriageResult(
        val probability: Float,
        val isSuspectCataract: Boolean,
        val latencyMs: Long,
        val preprocessedBitmap: Bitmap,
        val gradCamOverlay: Bitmap
    )

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val classifierWeights: FloatArray

    init {
        val modelBytes = context.assets.open("cataract_mobilenetv3.onnx").use { it.readBytes() }
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
        }
        session = env.createSession(modelBytes, opts)

        val weightsJson = context.assets.open("classifier_weights.json").bufferedReader().use { it.readText() }
        val jsonArr = JSONObject(weightsJson).getJSONArray("weights")
        classifierWeights = FloatArray(jsonArr.length()) { idx -> jsonArr.getDouble(idx).toFloat() }
    }

    fun analyze(bitmap: Bitmap): TriageResult {
        val prep = ImagePreprocessor.process(bitmap)
        val inputShape = longArrayOf(
            1L,
            3L,
            ImagePreprocessor.IMAGE_SIZE.toLong(),
            ImagePreprocessor.IMAGE_SIZE.toLong()
        )
        val inputName = session.inputNames.first()

        val t0 = System.nanoTime()
        val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(prep.tensorNchw), inputShape)
        val outputs = session.run(mapOf(inputName to inputTensor))
        val elapsedMs = (System.nanoTime() - t0) / 1_000_000L

        val logit = (outputs[0].value as FloatArray)[0]
        val convFeatures = (outputs[1].value as Array<Array<Array<FloatArray>>>)[0] // [C][H][W]
        outputs.close()
        inputTensor.close()

        val prob = (1.0f / (1.0f + exp(-logit)))
        val overlay = buildGradCamOverlay(prep.displayBitmap, convFeatures)

        return TriageResult(
            probability = prob,
            isSuspectCataract = prob >= 0.5f,
            latencyMs = elapsedMs,
            preprocessedBitmap = prep.displayBitmap,
            gradCamOverlay = overlay
        )
    }

    private fun buildGradCamOverlay(
        baseBitmap: Bitmap,
        convFeatures: Array<Array<FloatArray>>
    ): Bitmap {
        val channels = convFeatures.size
        val fh = convFeatures[0].size
        val fw = convFeatures[0][0].size
        val camGrid = Array(fh) { FloatArray(fw) }
        var maxVal = 0f

        for (y in 0 until fh) {
            for (x in 0 until fw) {
                var sum = 0f
                for (c in 0 until channels) {
                    sum += classifierWeights[c] * convFeatures[c][y][x]
                }
                val relu = max(0f, sum)
                camGrid[y][x] = relu
                if (relu > maxVal) maxVal = relu
            }
        }

        val heatmapSmall = Bitmap.createBitmap(fw, fh, Bitmap.Config.ARGB_8888)
        for (y in 0 until fh) {
            for (x in 0 until fw) {
                val norm = if (maxVal > 1e-8f) camGrid[y][x] / maxVal else 0f
                heatmapSmall.setPixel(x, y, jetColor(norm))
            }
        }
        val heatmapScaled = Bitmap.createScaledBitmap(
            heatmapSmall,
            ImagePreprocessor.IMAGE_SIZE,
            ImagePreprocessor.IMAGE_SIZE,
            true
        )

        val size = ImagePreprocessor.IMAGE_SIZE
        val basePixels = IntArray(size * size)
        val heatPixels = IntArray(size * size)
        baseBitmap.getPixels(basePixels, 0, size, 0, 0, size, size)
        heatmapScaled.getPixels(heatPixels, 0, size, 0, 0, size, size)

        for (i in basePixels.indices) {
            val b = basePixels[i]
            val h = heatPixels[i]
            val r = (((b shr 16) and 0xFF) * 0.55f + ((h shr 16) and 0xFF) * 0.45f).toInt().coerceIn(0, 255)
            val g = (((b shr 8) and 0xFF) * 0.55f + ((h shr 8) and 0xFF) * 0.45f).toInt().coerceIn(0, 255)
            val bl = ((b and 0xFF) * 0.55f + (h and 0xFF) * 0.45f).toInt().coerceIn(0, 255)
            basePixels[i] = Color.rgb(r, g, bl)
        }

        return Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888).apply {
            setPixels(basePixels, 0, size, 0, 0, size, size)
        }
    }

    private fun jetColor(v: Float): Int {
        val x = v.coerceIn(0f, 1f)
        val r = (1.5f - kotlin.math.abs(4f * x - 3f)).coerceIn(0f, 1f)
        val g = (1.5f - kotlin.math.abs(4f * x - 2f)).coerceIn(0f, 1f)
        val b = (1.5f - kotlin.math.abs(4f * x - 1f)).coerceIn(0f, 1f)
        return Color.rgb((r * 255).toInt(), (g * 255).toInt(), (b * 255).toInt())
    }

    override fun close() {
        session.close()
        env.close()
    }
}
