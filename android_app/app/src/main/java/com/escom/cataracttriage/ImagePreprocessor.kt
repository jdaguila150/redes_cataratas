package com.escom.cataracttriage

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Mirrors the Python preprocessing pipeline offline on Android:
 * 1. Central ocular ROI crop (90% of shorter side)
 * 2. Specular glare suppression (V > 235, S < 45 -> local 3x3 interpolation)
 * 3. Contrast enhancement on luminance (CLAHE-style tile histogram equalization)
 * 4. Resize to 224x224 and ImageNet normalization (NCHW FloatArray)
 */
object ImagePreprocessor {

    const val IMAGE_SIZE = 224
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    data class PreprocessedOutput(
        val displayBitmap: Bitmap,
        val tensorNchw: FloatArray
    )

    fun process(source: Bitmap): PreprocessedOutput {
        val roi = cropCentralOcularRoi(source)
        val resized = Bitmap.createScaledBitmap(roi, IMAGE_SIZE, IMAGE_SIZE, true)
        val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        resized.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)

        suppressGlareInPlace(pixels, IMAGE_SIZE, IMAGE_SIZE)
        enhanceLocalContrastInPlace(pixels, IMAGE_SIZE, IMAGE_SIZE)

        val displayBitmap = Bitmap.createBitmap(IMAGE_SIZE, IMAGE_SIZE, Bitmap.Config.ARGB_8888)
        displayBitmap.setPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)

        val nchw = FloatArray(3 * IMAGE_SIZE * IMAGE_SIZE)
        val planeSize = IMAGE_SIZE * IMAGE_SIZE
        for (i in 0 until planeSize) {
            val c = pixels[i]
            val r = ((c shr 16) and 0xFF) / 255.0f
            val g = ((c shr 8) and 0xFF) / 255.0f
            val b = (c and 0xFF) / 255.0f
            nchw[i] = (r - MEAN[0]) / STD[0]
            nchw[planeSize + i] = (g - MEAN[1]) / STD[1]
            nchw[2 * planeSize + i] = (b - MEAN[2]) / STD[2]
        }
        return PreprocessedOutput(displayBitmap, nchw)
    }

    private fun cropCentralOcularRoi(bitmap: Bitmap): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        val minDim = min(w, h)
        val cropSize = max(16, (minDim * 0.90f).roundToInt())
        val x1 = (w - cropSize) / 2
        val y1 = (h - cropSize) / 2
        return Bitmap.createBitmap(bitmap, x1, y1, cropSize, cropSize)
    }

    private fun suppressGlareInPlace(pixels: IntArray, w: Int, h: Int) {
        val hsv = FloatArray(3)
        val isGlare = BooleanArray(w * h)
        for (i in pixels.indices) {
            Color.colorToHSV(pixels[i], hsv)
            if (hsv[2] > 0.92f && hsv[1] < 0.18f) {
                isGlare[i] = true
            }
        }
        val copy = pixels.clone()
        for (y in 1 until h - 1) {
            for (x in 1 until w - 1) {
                val idx = y * w + x
                if (!isGlare[idx]) continue
                var rSum = 0
                var gSum = 0
                var bSum = 0
                var count = 0
                for (dy in -1..1) {
                    for (dx in -1..1) {
                        val nIdx = (y + dy) * w + (x + dx)
                        if (!isGlare[nIdx]) {
                            val c = copy[nIdx]
                            rSum += (c shr 16) and 0xFF
                            gSum += (c shr 8) and 0xFF
                            bSum += c and 0xFF
                            count++
                        }
                    }
                }
                if (count > 0) {
                    pixels[idx] = Color.rgb(rSum / count, gSum / count, bSum / count)
                }
            }
        }
    }

    private fun enhanceLocalContrastInPlace(pixels: IntArray, w: Int, h: Int) {
        // Simple luminance-preserving contrast stretch (clipped histogram equalization)
        val lum = IntArray(w * h)
        val hist = IntArray(256)
        for (i in pixels.indices) {
            val c = pixels[i]
            val r = (c shr 16) and 0xFF
            val g = (c shr 8) and 0xFF
            val b = c and 0xFF
            val y = ((77 * r + 150 * g + 29 * b) shr 8).coerceIn(0, 255)
            lum[i] = y
            hist[y]++
        }

        // Clip histogram (CLAHE-style global clip)
        val clipLimit = max(1, (w * h) / 64)
        var excess = 0
        for (b in 0 until 256) {
            if (hist[b] > clipLimit) {
                excess += hist[b] - clipLimit
                hist[b] = clipLimit
            }
        }
        val addPerBin = excess / 256
        for (b in 0 until 256) hist[b] += addPerBin

        val lut = IntArray(256)
        var cdf = 0
        val total = hist.sum().coerceAtLeast(1)
        for (b in 0 until 256) {
            cdf += hist[b]
            lut[b] = ((cdf * 255.0f) / total).roundToInt().coerceIn(0, 255)
        }

        for (i in pixels.indices) {
            val oldY = lum[i].coerceAtLeast(1)
            val newY = lut[lum[i]]
            val scale = newY.toFloat() / oldY.toFloat()
            val c = pixels[i]
            val r = (((c shr 16) and 0xFF) * scale).roundToInt().coerceIn(0, 255)
            val g = (((c shr 8) and 0xFF) * scale).roundToInt().coerceIn(0, 255)
            val b = ((c and 0xFF) * scale).roundToInt().coerceIn(0, 255)
            pixels[i] = Color.rgb(r, g, b)
        }
    }
}
