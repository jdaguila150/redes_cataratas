package com.escom.cataracttriage

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var engine: CataractTriageEngine
    private lateinit var txtStatus: TextView
    private lateinit var imgPreprocessed: ImageView
    private lateinit var imgGradCam: ImageView

    private val cameraLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicturePreview()
    ) { bitmap: Bitmap? ->
        bitmap?.let { runTriage(it) }
    }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted: Boolean ->
        if (granted) cameraLauncher.launch(null)
        else txtStatus.text = "Permiso de cámara denegado."
    }

    private val galleryLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { loadUriBitmap(it)?.let(::runTriage) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        txtStatus = findViewById(R.id.txtStatus)
        imgPreprocessed = findViewById(R.id.imgPreprocessed)
        imgGradCam = findViewById(R.id.imgGradCam)
        engine = CataractTriageEngine(this)

        findViewById<Button>(R.id.btnCamera).setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED
            ) {
                cameraLauncher.launch(null)
            } else {
                permissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }

        findViewById<Button>(R.id.btnGallery).setOnClickListener {
            galleryLauncher.launch("image/*")
        }
    }

    private fun loadUriBitmap(uri: Uri): Bitmap? = try {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val src = ImageDecoder.createSource(contentResolver, uri)
            ImageDecoder.decodeBitmap(src) { decoder, _, _ ->
                decoder.isMutableRequired = true
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            }
        } else {
            @Suppress("DEPRECATION")
            MediaStore.Images.Media.getBitmap(contentResolver, uri)
        }
    } catch (_: Exception) {
        txtStatus.text = "No se pudo cargar la imagen."
        null
    }

    private fun runTriage(bitmap: Bitmap) {
        txtStatus.text = "Analizando localmente..."
        lifecycleScope.launch {
            val res = withContext(Dispatchers.Default) { engine.analyze(bitmap) }
            val pct = (res.probability * 100f)
            val label = if (res.isSuspectCataract) {
                "SOSPECHA DE CATARATA — Canalizar a valoración oftalmológica"
            } else {
                "SIN SOSPECHA APARENTE (Normal)"
            }
            txtStatus.text = String.format(
                "%s\nProbabilidad: %.1f%% | Inferencia local: %d ms",
                label, pct, res.latencyMs
            )
            imgPreprocessed.setImageBitmap(res.preprocessedBitmap)
            imgGradCam.setImageBitmap(res.gradCamOverlay)
        }
    }

    override fun onDestroy() {
        engine.close()
        super.onDestroy()
    }
}
