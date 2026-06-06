package es.uc3m.android.pianotranscriber

import android.Manifest
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Base64
import android.util.Log
import android.util.Patterns
import android.widget.Toast
import android.widget.VideoView
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color as ComposeColor
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.window.Dialog
import androidx.core.content.ContextCompat
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import es.uc3m.android.pianotranscriber.ui.theme.AppTypography
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.asRequestBody
import org.json.JSONException
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.ConnectException
import java.net.SocketTimeoutException
import java.net.UnknownHostException
import java.text.SimpleDateFormat
import java.util.*

// Colour tokens
private val BG_DEEP    = ComposeColor(0xFF0F0F0F)
private val BG_CARD    = ComposeColor(0xFF1A1A1A)
private val BG_CHIP    = ComposeColor(0xFF252525)
private val ACCENT     = ComposeColor(0xFFA78BFA)
private val ACCENT_DIM = ComposeColor(0xFF2A1F4A)
private val SUCCESS    = ComposeColor(0xFF5DCAA5)
private val SUCCESS_DIM= ComposeColor(0xFF1A3D2A)
private val ERROR_CLR  = ComposeColor(0xFFE24B4A)
private val TXT_PRI    = ComposeColor(0xFFE5E5E5)
private val TXT_SEC    = ComposeColor(0xFF888888)
private val TXT_HINT   = ComposeColor(0xFF444444)
private val DIVIDER    = ComposeColor(0xFF2A2A2A)

// Minimum valid audio file size in bytes
private const val MIN_AUDIO_BYTES = 1024L

// DataStore for persisting the server URL
private val Context.settingsDataStore by preferencesDataStore(name = "settings")
private val SERVER_URL_KEY = stringPreferencesKey("server_url")

object Settings {
    fun serverUrlFlow(context: Context): Flow<String> =
        context.settingsDataStore.data.map { it[SERVER_URL_KEY] ?: Config.SERVER_URL }

    suspend fun currentServerUrl(context: Context): String =
        serverUrlFlow(context).first()

    suspend fun setServerUrl(context: Context, url: String) {
        context.settingsDataStore.edit { it[SERVER_URL_KEY] = url }
    }
}

// Data model
data class Recording(
    val id: String = UUID.randomUUID().toString(),
    val name: String,
    val durationSec: Int = 0,
    val timestamp: Long = System.currentTimeMillis(),
    val file: File? = null,
    val status: RecordingStatus = RecordingStatus.READY
)

enum class RecordingStatus { READY, PROCESSING, DONE, ERROR }
data class NoteEvent(val pitch: Int, val startBeat: Float, val durationBeats: Float)

// Server result
data class TranscriptionResult(
    val name: String,
    val midiFile: File,
    val videoFile: File,
    val bpm: Int
)

// App state
sealed class Screen {
    object Record       : Screen()
    object Processing   : Screen()
    object ResultList   : Screen()
    object ResultDetail : Screen()
}

data class ProcessingStep(val label: String, val state: StepState)
enum class StepState { DONE, ACTIVE, PENDING }

// Main Activity
class MainActivity : ComponentActivity() {

    private var recorder: AudioRecorder? = null

    // Currently in-flight upload job (so we can cancel on navigation away)
    private var uploadJob: Job? = null

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (!granted) Toast.makeText(this, "Microphone permission required", Toast.LENGTH_LONG).show()
    }

    private var onFilePicked: ((File) -> Unit)? = null
    private val filePicker = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        uri ?: return@registerForActivityResult
        try {
            val stream = contentResolver.openInputStream(uri)
            if (stream == null) {
                Toast.makeText(this, "Cannot read selected file", Toast.LENGTH_SHORT).show()
                return@registerForActivityResult
            }
            val dest = File(filesDir, "upload_${System.currentTimeMillis()}.wav")
            stream.use { it.copyTo(dest.outputStream()) }
            if (!dest.exists() || dest.length() < MIN_AUDIO_BYTES) {
                Toast.makeText(this, "Selected file is empty or too short", Toast.LENGTH_SHORT).show()
                runCatching { dest.delete() }
                return@registerForActivityResult
            }
            onFilePicked?.invoke(dest)
        } catch (e: SecurityException) {
            Log.e("FilePicker", "Permission denied reading file", e)
            Toast.makeText(this, "Permission denied reading file", Toast.LENGTH_SHORT).show()
        } catch (e: IOException) {
            Log.e("FilePicker", "I/O error copying file", e)
            Toast.makeText(this, "Could not import file (I/O error)", Toast.LENGTH_LONG).show()
        } catch (e: Exception) {
            Log.e("FilePicker", "Unexpected error importing file", e)
            Toast.makeText(this, "Could not import file: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestPermissionIfNeeded()
        setContent {
            PianoTranscriberTheme {
                MainScreen(
                    onStartRecording = { file ->
                        if (!hasMicPermission()) {
                            Toast.makeText(this, "Microphone permission required", Toast.LENGTH_LONG).show()
                            permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                            false
                        } else {
                            recorder = AudioRecorder(this, file)
                            val ok = recorder?.start() ?: false
                            if (!ok) {
                                Toast.makeText(
                                    this,
                                    "Could not start recording — microphone may be in use",
                                    Toast.LENGTH_LONG
                                ).show()
                                recorder = null
                            }
                            ok
                        }
                    },
                    onStopRecording  = {
                        try {
                            recorder?.stop() ?: 0
                        } catch (e: Exception) {
                            Log.e("MainActivity", "Error stopping recorder", e)
                            0
                        }
                    },
                    onPickFile       = { callback -> onFilePicked = callback; filePicker.launch("audio/*") },
                    onUpload         = { file, onProgress, onDone, onError ->
                        uploadAudio(file, onProgress, onDone, onError)
                    },
                    onCancelUpload   = { cancelUpload() },
                    onSaveMidi       = { saveMidiToDownloads(it) },
                    onSaveVideo      = { saveVideoToDownloads(it) },
                    onLifecyclePause = {
                        // Stop any active recording when the app goes to background.
                        // This protects against the user backgrounding mid-record and
                        // leaking the mic / leaving an unfinished file.
                        recorder?.let { rec ->
                            if (rec.isActive()) {
                                try { rec.stop() } catch (e: Exception) {
                                    Log.e("MainActivity", "Error stopping recorder on pause", e)
                                }
                            }
                        }
                        recorder = null
                    }
                )
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Defensive: ensure no recorder leaks if the Activity is destroyed
        // before the composable's DisposableEffect fires.
        recorder?.let {
            try { if (it.isActive()) it.stop() } catch (_: Exception) {}
        }
        recorder = null
        cancelUpload()
    }

    private fun cancelUpload() {
        uploadJob?.let { job ->
            if (job.isActive) {
                Log.d("Upload", "Cancelling in-flight upload")
                job.cancel()
            }
        }
        uploadJob = null
    }

    private fun hasMicPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED

    private fun requestPermissionIfNeeded() {
        if (!hasMicPermission()) permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
    }

    private fun uploadAudio(
        file: File,
        onProgress: (List<ProcessingStep>) -> Unit,
        onDone: (TranscriptionResult) -> Unit,
        onError: (String) -> Unit
    ) {
        if (!file.exists()) {
            onError("Recording file not found")
            return
        }
        if (file.length() < MIN_AUDIO_BYTES) {
            onError("Recording too short or empty")
            return
        }

        // Cancel any previous upload before starting a new one
        cancelUpload()

        uploadJob = CoroutineScope(Dispatchers.IO).launch {
            val steps = mutableListOf(
                ProcessingStep("Transcribing…",     StepState.PENDING),
                ProcessingStep("Preparing results", StepState.PENDING),
            )
            fun resetStepsAndReport() {
                val reset = steps.map { it.copy(state = StepState.PENDING) }
                runOnUiThread { onProgress(reset) }
            }
            fun reportError(msg: String) {
                resetStepsAndReport()
                runOnUiThread { onError(msg) }
            }

            // Resolve the current server URL from DataStore (falls back to Config.SERVER_URL)
            val serverUrl = try {
                Settings.currentServerUrl(this@MainActivity)
            } catch (e: Exception) {
                Log.e("Upload", "Could not read server URL setting, using default", e)
                Config.SERVER_URL
            }

            try {
                fun advance(idx: Int) {
                    for (i in steps.indices) {
                        steps[i] = steps[i].copy(state = when {
                            i < idx  -> StepState.DONE
                            i == idx -> StepState.ACTIVE
                            else     -> StepState.PENDING
                        })
                    }
                    runOnUiThread { onProgress(steps.toList()) }
                }
                val client = OkHttpClient.Builder()
                    .connectTimeout(60,  java.util.concurrent.TimeUnit.SECONDS)
                    .readTimeout(300,    java.util.concurrent.TimeUnit.SECONDS)
                    .writeTimeout(60,    java.util.concurrent.TimeUnit.SECONDS)
                    .build()
                advance(0)
                val requestBody = MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart("file", file.name, file.asRequestBody("audio/*".toMediaType()))
                    .build()
                val request  = Request.Builder().url(serverUrl).post(requestBody).build()
                val call = client.newCall(request)

                // Tie OkHttp call cancellation to coroutine cancellation
                val response = try {
                    call.execute()
                } catch (e: IOException) {
                    if (!isActive) {
                        Log.d("Upload", "Upload cancelled during request")
                        return@launch
                    }
                    throw e
                }

                if (!isActive) {
                    response.close()
                    call.cancel()
                    return@launch
                }

                if (!response.isSuccessful) {
                    val msg = when (response.code) {
                        in 500..599 -> "Server error (${response.code}) — please try again"
                        413          -> "Recording too large for server"
                        408, 504     -> "Server timed out — try a shorter clip"
                        else         -> "Server error ${response.code}"
                    }
                    reportError(msg)
                    response.close()
                    return@launch
                }
                advance(1)
                delay(5000)

                val body = response.body?.string()
                if (body.isNullOrBlank()) {
                    reportError("Empty response from server")
                    return@launch
                }

                val json = try {
                    JSONObject(body)
                } catch (e: JSONException) {
                    Log.e("Upload", "Invalid JSON in response", e)
                    reportError("Invalid response from server")
                    return@launch
                }

                val bpm = try {
                    json.getDouble("bpm").toInt()
                } catch (e: JSONException) {
                    Log.e("Upload", "Missing bpm field", e)
                    reportError("Server response missing tempo")
                    return@launch
                }
                val name = file.nameWithoutExtension

                val midiFile = try {
                    val midiB64 = json.getString("midi")
                    val midiBytes = Base64.decode(midiB64, Base64.DEFAULT)
                    if (midiBytes.isEmpty()) throw IOException("Empty MIDI payload")
                    val f = File(filesDir, "result_${System.currentTimeMillis()}.mid")
                    f.writeBytes(midiBytes)
                    f
                } catch (e: JSONException) {
                    Log.e("Upload", "Missing midi field", e)
                    reportError("Server response missing MIDI data")
                    return@launch
                } catch (e: IllegalArgumentException) {
                    Log.e("Upload", "Corrupt Base64 in MIDI payload", e)
                    reportError("Corrupt MIDI in server response")
                    return@launch
                } catch (e: IOException) {
                    Log.e("Upload", "Could not write MIDI file", e)
                    reportError("Could not save MIDI (storage full?)")
                    return@launch
                }

                val videoFile = try {
                    val videoB64 = json.getString("video")
                    val videoBytes = Base64.decode(videoB64, Base64.DEFAULT)
                    if (videoBytes.isEmpty()) throw IOException("Empty video payload")
                    val f = File(filesDir, "result_${System.currentTimeMillis()}.mp4")
                    f.writeBytes(videoBytes)
                    f
                } catch (e: JSONException) {
                    Log.e("Upload", "Missing video field", e)
                    reportError("Server response missing video data")
                    return@launch
                } catch (e: IllegalArgumentException) {
                    Log.e("Upload", "Corrupt Base64 in video payload", e)
                    reportError("Corrupt video in server response")
                    return@launch
                } catch (e: IOException) {
                    Log.e("Upload", "Could not write video file", e)
                    reportError("Could not save video (storage full?)")
                    return@launch
                }

                if (!isActive) return@launch

                val finalSteps = steps.map { it.copy(state = StepState.DONE) }
                runOnUiThread { onProgress(finalSteps) }
                runOnUiThread { onDone(TranscriptionResult(name, midiFile, videoFile, bpm)) }
                delay(1000)
            } catch (e: UnknownHostException) {
                Log.e("Upload", "No internet / DNS failure", e)
                reportError("No internet connection")
            } catch (e: ConnectException) {
                Log.e("Upload", "Could not connect to server", e)
                reportError("Cannot reach server, is it running?")
            } catch (e: SocketTimeoutException) {
                Log.e("Upload", "Request timed out", e)
                reportError("Server timed out, try a shorter clip")
            } catch (e: IOException) {
                Log.e("Upload", "Network I/O error", e)
                reportError("Network error: ${e.message ?: "connection failed"}")
            } catch (e: CancellationException) {
                Log.d("Upload", "Upload cancelled")
                // Don't surface cancellation as an error to the user; the UI has
                // already navigated away. Still need to reset internal step state.
                runOnUiThread {
                    val reset = steps.map { it.copy(state = StepState.PENDING) }
                    onProgress(reset)
                }
                throw e
            } catch (e: Exception) {
                Log.e("Upload", "Upload failed", e)
                reportError(e.message ?: "Unknown error")
            }
        }
    }

    // Save MIDI
    private fun saveMidiToDownloads(midiFile: File) {
        if (!midiFile.exists()) {
            Toast.makeText(this, "MIDI file not found", Toast.LENGTH_SHORT).show()
            return
        }
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) saveMidiApi29(midiFile)
            else saveMidiLegacy(midiFile)
        } catch (e: Exception) {
            Log.e("SaveMidi", "Save failed", e)
            Toast.makeText(this, "Save failed: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    private fun saveMidiApi29(midiFile: File) {
        val values = ContentValues().apply {
            put(MediaStore.Downloads.DISPLAY_NAME, midiFile.name)
            put(MediaStore.Downloads.MIME_TYPE, "audio/midi")
            put(MediaStore.Downloads.IS_PENDING, 1)
        }
        val uri = contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
        if (uri == null) {
            Toast.makeText(this, "Could not create Downloads entry", Toast.LENGTH_SHORT).show()
            return
        }
        try {
            contentResolver.openOutputStream(uri)?.use { midiFile.inputStream().copyTo(it) }
                ?: throw IOException("Could not open output stream")
            values.clear(); values.put(MediaStore.Downloads.IS_PENDING, 0)
            contentResolver.update(uri, values, null, null)
            Toast.makeText(this, "MIDI saved to Downloads", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            runCatching { contentResolver.delete(uri, null, null) }
            throw e
        }
    }

    private fun saveMidiLegacy(midiFile: File) {
        try {
            val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            dir.mkdirs()
            midiFile.inputStream().use { i -> FileOutputStream(File(dir, midiFile.name)).use { o -> i.copyTo(o) } }
            Toast.makeText(this, "MIDI saved to Downloads/${midiFile.name}", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) { Toast.makeText(this, "Save failed: ${e.message}", Toast.LENGTH_LONG).show() }
    }

    // Save Video
    private fun saveVideoToDownloads(videoFile: File) {
        if (!videoFile.exists()) {
            Toast.makeText(this, "Video file not found", Toast.LENGTH_SHORT).show()
            return
        }
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) saveVideoApi29(videoFile)
            else saveVideoLegacy(videoFile)
        } catch (e: Exception) {
            Log.e("SaveVideo", "Save failed", e)
            Toast.makeText(this, "Save failed: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    private fun saveVideoApi29(videoFile: File) {
        val values = ContentValues().apply {
            put(MediaStore.Downloads.DISPLAY_NAME, videoFile.name)
            put(MediaStore.Downloads.MIME_TYPE, "video/mp4")
            put(MediaStore.Downloads.IS_PENDING, 1)
        }
        val uri = contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
        if (uri == null) {
            Toast.makeText(this, "Could not create Downloads entry", Toast.LENGTH_SHORT).show()
            return
        }
        try {
            contentResolver.openOutputStream(uri)?.use { videoFile.inputStream().copyTo(it) }
                ?: throw IOException("Could not open output stream")
            values.clear(); values.put(MediaStore.Downloads.IS_PENDING, 0)
            contentResolver.update(uri, values, null, null)
            Toast.makeText(this, "Video saved to Downloads", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            runCatching { contentResolver.delete(uri, null, null) }
            throw e
        }
    }

    private fun saveVideoLegacy(videoFile: File) {
        try {
            val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            dir.mkdirs()
            videoFile.inputStream().use { i -> FileOutputStream(File(dir, videoFile.name)).use { o -> i.copyTo(o) } }
            Toast.makeText(this, "Video saved to Downloads/${videoFile.name}", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) { Toast.makeText(this, "Save failed: ${e.message}", Toast.LENGTH_LONG).show() }
    }
}

// Theme
@Composable
fun PianoTranscriberTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(
            background = BG_DEEP, surface = BG_CARD, primary = ACCENT,
            onPrimary = ComposeColor.White, onBackground = TXT_PRI, onSurface = TXT_PRI,
        ),
        typography = AppTypography, content = content
    )
}

// Main screen
@Composable
fun MainScreen(
    onStartRecording: (File) -> Boolean,
    onStopRecording: () -> Int,
    onPickFile: (callback: (File) -> Unit) -> Unit,
    onUpload: (File, (List<ProcessingStep>) -> Unit, (TranscriptionResult) -> Unit, (String) -> Unit) -> Unit,
    onCancelUpload: () -> Unit,
    onSaveMidi:  (File) -> Unit,
    onSaveVideo: (File) -> Unit,
    onLifecyclePause: () -> Unit
) {
    val context = LocalContext.current
    var screen          by remember { mutableStateOf<Screen>(Screen.Record) }
    var isRecording     by remember { mutableStateOf(false) }
    var recSeconds      by remember { mutableIntStateOf(0) }
    var recordings      by remember { mutableStateOf(listOf<Recording>()) }
    var processingSteps by remember { mutableStateOf(initialSteps()) }
    var results         by remember { mutableStateOf(listOf<TranscriptionResult>()) }
    var selectedResult  by remember { mutableStateOf<TranscriptionResult?>(null) }
    var currentFile     by remember { mutableStateOf<File?>(null) }
    var currentRecFile  by remember { mutableStateOf<File?>(null) }
    var showSettings    by remember { mutableStateOf(false) }

    // Lifecycle hook: stop recording when app goes to background
    val lifecycleOwner = LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_PAUSE && isRecording) {
                onLifecyclePause()
                isRecording = false
                currentRecFile = null
                Toast.makeText(
                    context,
                    "Recording stopped (app backgrounded)",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }

    val renameResult = { res: TranscriptionResult, newName: String ->
        val renamed = res.copy(name = newName)
        results = results.map { if (it === res) renamed else it }
        if (selectedResult === res) selectedResult = renamed
    }
    val deleteResult = { res: TranscriptionResult ->
        results = results.filter { it !== res }
        if (selectedResult === res) { selectedResult = null; screen = Screen.ResultList }
    }

    LaunchedEffect(isRecording) {
        if (isRecording) {
            recSeconds = 0
            while (isRecording) { delay(1000); recSeconds++ }
        } else {
            recSeconds = 0
        }
    }

    fun startUpload(file: File, name: String) {
        if (!file.exists() || file.length() < MIN_AUDIO_BYTES) {
            Toast.makeText(context, "Recording too short or empty", Toast.LENGTH_SHORT).show()
            return
        }
        currentFile = file
        recordings  = listOf(Recording(name = name, file = file, status = RecordingStatus.PROCESSING)) + recordings
        processingSteps = initialSteps()
        screen = Screen.Processing
        onUpload(file,
            { steps -> processingSteps = steps },
            { res ->
                results = listOf(res) + results
                selectedResult = res
                recordings = recordings.map { if (it.name == name) it.copy(status = RecordingStatus.DONE) else it }
                currentFile = null
                // Only jump to the result if the user is still waiting on the
                // Processing screen. If they navigated away while the upload kept
                // running in the background, don't yank them, the finished
                // transcription is available in the Results tab.
                if (screen == Screen.Processing) {
                    screen = Screen.ResultDetail
                } else {
                    Toast.makeText(context, "Transcription ready — see Results", Toast.LENGTH_SHORT).show()
                }
                // Reset the Processing screen back to its idle state so it
                // doesn't stay stuck on the green "complete" view next time
                // the user opens the Processing tab.
                processingSteps = initialSteps()
            },
            { err ->
                Toast.makeText(context, "Error: $err", Toast.LENGTH_LONG).show()
                recordings = recordings.map { if (it.name == name) it.copy(status = RecordingStatus.ERROR) else it }
                processingSteps = initialSteps()
                currentFile = null
                // Only bounce back to Record if the user is still on Processing.
                // If they've navigated elsewhere, leave them where they are.
                if (screen == Screen.Processing) {
                    screen = Screen.Record
                }
            }
        )
    }

    // Switch tabs. Any in-flight upload is intentionally left running in the
    // background so the transcription keeps going when the user navigates away;
    // the result lands in the Results tab (and opens automatically if they're
    // still waiting on the Processing screen) when it finishes.
    fun navigateAwayFrom(target: Screen) {
        screen = target
    }

    // Settings dialog
    if (showSettings) {
        ServerSettingsDialog(
            onDismiss = { showSettings = false },
            onSaved   = {
                showSettings = false
                Toast.makeText(context, "Server URL saved", Toast.LENGTH_SHORT).show()
            }
        )
    }

    Column(modifier = Modifier.fillMaxSize().background(BG_DEEP).windowInsetsPadding(WindowInsets.systemBars)) {
        Row(modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 14.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            if (screen == Screen.ResultDetail) {
                Icon(Icons.Default.ArrowBack, contentDescription = "Back", tint = ACCENT,
                    modifier = Modifier.size(20.dp).clickable(
                        interactionSource = remember { MutableInteractionSource() }, indication = null
                    ) { screen = Screen.ResultList })
            } else {
                Icon(Icons.Default.PlayArrow, contentDescription = null, tint = ACCENT, modifier = Modifier.size(20.dp))
            }
            Spacer(Modifier.width(8.dp))
            Text("Piano Transcriber", style = MaterialTheme.typography.titleMedium, color = TXT_PRI)
            Spacer(Modifier.weight(1f))
            Spacer(Modifier.width(8.dp))
            Icon(Icons.Default.Settings, contentDescription = "Settings", tint = TXT_SEC,
                modifier = Modifier.size(20.dp).clickable(
                    interactionSource = remember { MutableInteractionSource() }, indication = null
                ) { showSettings = true })
        }
        NavTabs(screen) { newScreen -> navigateAwayFrom(newScreen) }
        HorizontalDivider(color = DIVIDER, thickness = 0.5.dp)
        Box(Modifier.fillMaxSize()) {
            when (screen) {
                Screen.Record -> RecordScreen(
                    isRecording = isRecording, recordingSeconds = recSeconds, recordings = recordings,
                    onToggleRecord = { filesDir ->
                        if (!isRecording) {
                            val f = File(filesDir, "rec_${System.currentTimeMillis()}.m4a")
                            currentRecFile = f
                            val started = onStartRecording(f)
                            if (started) {
                                isRecording = true
                            } else {
                                currentRecFile = null
                            }
                        } else {
                            val duration = onStopRecording()
                            isRecording = false
                            val file = currentRecFile
                            if (file != null && duration > 0 && file.exists() && file.length() >= MIN_AUDIO_BYTES) {
                                startUpload(file, "Recording ${recordings.size + 1}")
                            } else {
                                Toast.makeText(
                                    context,
                                    "Recording too short — please hold for at least 1 second",
                                    Toast.LENGTH_SHORT
                                ).show()
                                runCatching { file?.delete() }
                            }
                            currentRecFile = null
                        }
                    },
                    onUploadFile = { onPickFile { f -> startUpload(f, f.nameWithoutExtension) } },
                    onResubmit   = { rec -> rec.file?.let { startUpload(it, rec.name) } }
                )
                Screen.Processing -> ProcessingScreen(
                    steps    = processingSteps,
                    fileName = currentFile?.name ?: "",
                    isDone   = processingSteps.all { it.state == StepState.DONE }
                )
                Screen.ResultList -> ResultListScreen(
                    results  = results,
                    onSelect = { res -> selectedResult = res; screen = Screen.ResultDetail },
                    onRename = renameResult, onDelete = deleteResult
                )
                Screen.ResultDetail -> Box(modifier = Modifier.fillMaxSize().background(BG_DEEP)) {
                    ResultScreen(
                        result         = selectedResult,
                        onExportMidi   = { selectedResult?.midiFile?.let  { onSaveMidi(it) } },
                        onExportVideo  = { selectedResult?.videoFile?.let { onSaveVideo(it) } },
                        onNewRecording = { screen = Screen.Record }
                    )
                }
            }
        }
    }
}

fun initialSteps() = listOf(
    ProcessingStep("Transcribing…",     StepState.PENDING),
    ProcessingStep("Preparing results", StepState.PENDING),
)

// Settings dialog
@Composable
fun ServerSettingsDialog(onDismiss: () -> Unit, onSaved: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var urlText by remember { mutableStateOf("") }
    var loaded by remember { mutableStateOf(false) }
    var errorText by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(Unit) {
        urlText = try {
            Settings.currentServerUrl(context)
        } catch (e: Exception) {
            Config.SERVER_URL
        }
        loaded = true
    }

    fun isValidUrl(s: String): Boolean {
        val trimmed = s.trim()
        if (trimmed.isEmpty()) return false
        if (!trimmed.startsWith("http://") && !trimmed.startsWith("https://")) return false
        return Patterns.WEB_URL.matcher(trimmed).matches()
    }

    Dialog(onDismissRequest = onDismiss) {
        Column(
            modifier = Modifier.fillMaxWidth()
                .background(BG_CARD, RoundedCornerShape(12.dp))
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            Text("Server settings", fontSize = 14.sp, fontWeight = FontWeight.Medium, color = TXT_PRI)
            Text(
                "Backend URL — point this at your FastAPI server",
                fontSize = 11.sp, color = TXT_HINT
            )
            BasicTextField(
                value = urlText,
                onValueChange = { urlText = it; errorText = null },
                singleLine = true,
                textStyle = androidx.compose.ui.text.TextStyle(
                    color = TXT_PRI, fontSize = 12.sp, fontFamily = FontFamily.Monospace
                ),
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                keyboardActions = KeyboardActions(onDone = {
                    if (isValidUrl(urlText)) {
                        scope.launch {
                            Settings.setServerUrl(context, urlText.trim())
                            onSaved()
                        }
                    } else {
                        errorText = "Must be http:// or https:// URL"
                    }
                }),
                modifier = Modifier.fillMaxWidth()
                    .background(BG_CHIP, RoundedCornerShape(8.dp))
                    .border(0.5.dp, if (errorText != null) ERROR_CLR else ACCENT, RoundedCornerShape(8.dp))
                    .padding(horizontal = 12.dp, vertical = 10.dp)
            )
            if (errorText != null) {
                Text(errorText!!, fontSize = 10.sp, color = ERROR_CLR)
            }
            Text(
                "Tip: use http://<your-laptop-ip>:8000/transcribe when on the same network",
                fontSize = 10.sp, color = TXT_HINT
            )
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinedButton(
                    onClick = onDismiss,
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = TXT_SEC),
                    border = androidx.compose.foundation.BorderStroke(0.5.dp, DIVIDER),
                    shape = RoundedCornerShape(8.dp)
                ) { Text("Cancel", fontSize = 12.sp) }
                OutlinedButton(
                    onClick = {
                        urlText = Config.SERVER_URL
                        errorText = null
                    },
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = TXT_SEC),
                    border = androidx.compose.foundation.BorderStroke(0.5.dp, DIVIDER),
                    shape = RoundedCornerShape(8.dp),
                    enabled = loaded
                ) { Text("Default", fontSize = 12.sp) }
                Button(
                    onClick = {
                        if (isValidUrl(urlText)) {
                            scope.launch {
                                Settings.setServerUrl(context, urlText.trim())
                                onSaved()
                            }
                        } else {
                            errorText = "Must be http:// or https:// URL"
                        }
                    },
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(containerColor = ACCENT_DIM, contentColor = ACCENT),
                    shape = RoundedCornerShape(8.dp),
                    enabled = loaded
                ) { Text("Save", fontSize = 12.sp) }
            }
        }
    }
}

// Nav tabs
@Composable
fun NavTabs(current: Screen, onSelect: (Screen) -> Unit) {
    val tabs = listOf("Record" to Screen.Record, "Processing" to Screen.Processing, "Results" to Screen.ResultList)
    Row(Modifier.fillMaxWidth()) {
        tabs.forEach { (label, scr) ->
            val active = current == scr || (scr == Screen.ResultList && current == Screen.ResultDetail)
            Box(modifier = Modifier.weight(1f)
                .clickable(interactionSource = remember { MutableInteractionSource() }, indication = null) { onSelect(scr) }
                .padding(vertical = 10.dp), contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(label, fontSize = 12.sp, color = if (active) ACCENT else TXT_HINT,
                        fontWeight = if (active) FontWeight.Medium else FontWeight.Normal)
                    Spacer(Modifier.height(6.dp))
                    Box(modifier = Modifier.height(2.dp).width(if (active) 32.dp else 0.dp).background(ACCENT, RoundedCornerShape(1.dp)))
                }
            }
        }
    }
}

// Record screen
@Composable
fun RecordScreen(
    isRecording: Boolean, recordingSeconds: Int, recordings: List<Recording>,
    onToggleRecord: (java.io.File) -> Unit, onUploadFile: () -> Unit, onResubmit: (Recording) -> Unit
) {
    val context = LocalContext.current
    val infiniteTransition = rememberInfiniteTransition(label = "wave")
    val barHeights = (0..11).map { i ->
        infiniteTransition.animateFloat(
            initialValue = 6f, targetValue = if (isRecording) 36f else 8f,
            animationSpec = infiniteRepeatable(
                animation = tween(300 + i * 40, easing = FastOutSlowInEasing),
                repeatMode = RepeatMode.Reverse, initialStartOffset = StartOffset(i * 30)
            ), label = "bar$i"
        )
    }
    LazyColumn(modifier = Modifier.fillMaxSize().padding(horizontal = 20.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp), contentPadding = PaddingValues(vertical = 20.dp)
    ) {
        item {
            Box(modifier = Modifier.fillMaxWidth().height(80.dp)
                .background(BG_CARD, RoundedCornerShape(12.dp)).border(0.5.dp, DIVIDER, RoundedCornerShape(12.dp)),
                contentAlignment = Alignment.Center
            ) {
                Row(horizontalArrangement = Arrangement.spacedBy(5.dp), verticalAlignment = Alignment.CenterVertically) {
                    barHeights.forEachIndexed { i, anim ->
                        Box(modifier = Modifier.width(3.dp)
                            .height((if (isRecording) anim.value else 6f + (i % 3) * 4f).dp)
                            .background(if (isRecording) ACCENT else TXT_HINT, RoundedCornerShape(2.dp)))
                    }
                }
            }
        }
        item {
            Text(text = "%02d:%02d".format(recordingSeconds / 60, recordingSeconds % 60),
                fontSize = 28.sp, fontWeight = FontWeight.Medium, fontFamily = FontFamily.Monospace,
                color = if (isRecording) ERROR_CLR else TXT_PRI, modifier = Modifier.fillMaxWidth(),
                textAlign = androidx.compose.ui.text.style.TextAlign.Center)
        }
        item {
            val pulseAnim = rememberInfiniteTransition(label = "pulse")
            val scale by pulseAnim.animateFloat(
                initialValue = 1f, targetValue = if (isRecording) 1.12f else 1f,
                animationSpec = infiniteRepeatable(tween(900), RepeatMode.Reverse), label = "s"
            )
            Box(modifier = Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
                Box(modifier = Modifier.size((68 * scale).dp)
                    .background(if (isRecording) ERROR_CLR else ACCENT, CircleShape)
                    .clickable(interactionSource = remember { MutableInteractionSource() }, indication = null)
                    { onToggleRecord(context.filesDir) }, contentAlignment = Alignment.Center
                ) {
                    Box(modifier = Modifier.size(22.dp).background(ComposeColor.White,
                        if (isRecording) RoundedCornerShape(3.dp) else CircleShape))
                }
            }
        }
        item {
            Text(if (isRecording) "Tap to stop and transcribe" else "Tap to start recording",
                fontSize = 12.sp, color = TXT_SEC, modifier = Modifier.fillMaxWidth(),
                textAlign = androidx.compose.ui.text.style.TextAlign.Center)
        }
        item {
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                HorizontalDivider(Modifier.weight(1f), color = DIVIDER, thickness = 0.5.dp)
                Text("  or  ", fontSize = 11.sp, color = TXT_HINT)
                HorizontalDivider(Modifier.weight(1f), color = DIVIDER, thickness = 0.5.dp)
            }
        }
        item {
            Box(modifier = Modifier.fillMaxWidth().border(0.5.dp, TXT_HINT, RoundedCornerShape(10.dp))
                .background(BG_CARD, RoundedCornerShape(10.dp))
                .clickable(interactionSource = remember { MutableInteractionSource() }, indication = null) { onUploadFile() }
                .padding(12.dp), contentAlignment = Alignment.Center
            ) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.Add, contentDescription = null, tint = TXT_SEC, modifier = Modifier.size(16.dp))
                    Text("Upload audio file (.wav)", fontSize = 13.sp, color = TXT_SEC)
                }
            }
        }
        if (recordings.isNotEmpty()) {
            item { Text("Recent recordings", fontSize = 11.sp, color = TXT_HINT, modifier = Modifier.padding(top = 4.dp)) }
            items(recordings) { rec -> RecordingItem(rec, onResubmit) }
        }
    }
}

@Composable
fun RecordingItem(rec: Recording, onResubmit: (Recording) -> Unit) {
    val time = SimpleDateFormat("HH:mm", Locale.getDefault()).format(Date(rec.timestamp))
    Row(modifier = Modifier.fillMaxWidth().background(BG_CARD, RoundedCornerShape(10.dp))
        .border(0.5.dp, DIVIDER, RoundedCornerShape(10.dp))
        .clickable(enabled = rec.status == RecordingStatus.DONE,
            interactionSource = remember { MutableInteractionSource() }, indication = null) { onResubmit(rec) }
        .padding(horizontal = 14.dp, vertical = 12.dp),
        horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically
    ) {
        Column {
            Text(rec.name, fontSize = 13.sp, color = TXT_PRI)
            Spacer(Modifier.height(3.dp))
            Text("%d:%02d  ·  %s".format(rec.durationSec / 60, rec.durationSec % 60, time),
                fontSize = 10.sp, color = TXT_HINT, fontFamily = FontFamily.Monospace)
        }
        StatusChip(rec.status)
    }
}

@Composable
fun StatusChip(status: RecordingStatus) {
    val (label, bg, fg) = when (status) {
        RecordingStatus.DONE       -> Triple("Done",       SUCCESS_DIM,             SUCCESS)
        RecordingStatus.PROCESSING -> Triple("Processing", ACCENT_DIM,              ACCENT)
        RecordingStatus.ERROR      -> Triple("Error",      ComposeColor(0xFF3D1A1A),ERROR_CLR)
        RecordingStatus.READY      -> Triple("Ready",      BG_CHIP,                 TXT_SEC)
    }
    Text(label, fontSize = 9.sp, fontWeight = FontWeight.Medium, color = fg,
        modifier = Modifier.background(bg, RoundedCornerShape(20.dp)).padding(horizontal = 8.dp, vertical = 3.dp))
}

// Processing screen
@Composable
fun ProcessingScreen(steps: List<ProcessingStep>, fileName: String, isDone: Boolean) {
    val isProcessing = steps.any { it.state == StepState.ACTIVE }
    val infiniteTransition = rememberInfiniteTransition(label = "spin")
    val rotation by infiniteTransition.animateFloat(
        initialValue = 0f, targetValue = 360f,
        animationSpec = infiniteRepeatable(tween(800, easing = LinearEasing)), label = "rot"
    )
    Column(modifier = Modifier.fillMaxSize().padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Spacer(Modifier.height(24.dp))
        Canvas(modifier = Modifier.size(48.dp)) {
            drawArc(color = DIVIDER, startAngle = 0f, sweepAngle = 360f, useCenter = false,
                style = androidx.compose.ui.graphics.drawscope.Stroke(width = 2.dp.toPx()))
            when {
                isDone       -> drawArc(color = SUCCESS, startAngle = 0f, sweepAngle = 360f, useCenter = false,
                    style = androidx.compose.ui.graphics.drawscope.Stroke(width = 2.dp.toPx()))
                isProcessing -> drawArc(color = ACCENT, startAngle = rotation, sweepAngle = 90f, useCenter = false,
                    style = androidx.compose.ui.graphics.drawscope.Stroke(width = 2.dp.toPx()))
            }
        }
        Text(
            when { isDone -> "Transcription complete"; isProcessing -> "Transcribing melody…"; else -> "No file selected" },
            fontSize = 14.sp, fontWeight = FontWeight.Medium, color = if (isDone) SUCCESS else TXT_PRI
        )
        Text(fileName, fontSize = 11.sp, color = TXT_HINT, fontFamily = FontFamily.Monospace)
        Spacer(Modifier.height(8.dp))
        steps.forEach { ProcessingStepRow(it) }
    }
}

@Composable
fun ProcessingStepRow(step: ProcessingStep) {
    val pulse = rememberInfiniteTransition(label = "dot")
    val alpha by pulse.animateFloat(
        initialValue = 1f, targetValue = if (step.state == StepState.ACTIVE) 0.3f else 1f,
        animationSpec = infiniteRepeatable(tween(900), RepeatMode.Reverse), label = "a"
    )
    val dotColor  = when (step.state) { StepState.DONE -> SUCCESS; StepState.ACTIVE -> ACCENT.copy(alpha = alpha); StepState.PENDING -> TXT_HINT }
    val textColor = when (step.state) { StepState.DONE -> SUCCESS; StepState.ACTIVE -> TXT_PRI; StepState.PENDING -> TXT_HINT }
    Row(modifier = Modifier.fillMaxWidth().background(BG_CARD, RoundedCornerShape(8.dp))
        .border(0.5.dp, DIVIDER, RoundedCornerShape(8.dp)).padding(horizontal = 14.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Box(modifier = Modifier.size(6.dp).background(dotColor, CircleShape))
        Text(step.label, fontSize = 12.sp, color = textColor, modifier = Modifier.weight(1f))
        if (step.state == StepState.DONE)
            Icon(Icons.Default.Check, contentDescription = null, tint = SUCCESS, modifier = Modifier.size(14.dp))
    }
}

// Result list screen
@Composable
fun ResultListScreen(
    results: List<TranscriptionResult>, onSelect: (TranscriptionResult) -> Unit,
    onRename: (TranscriptionResult, String) -> Unit, onDelete: (TranscriptionResult) -> Unit
) {
    var renamingResult by remember { mutableStateOf<TranscriptionResult?>(null) }
    var renameText     by remember { mutableStateOf("") }

    renamingResult?.let { res ->
        Dialog(onDismissRequest = { renamingResult = null }) {
            Column(modifier = Modifier.fillMaxWidth().background(BG_CARD, RoundedCornerShape(12.dp)).padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Text("Rename", fontSize = 14.sp, fontWeight = FontWeight.Medium, color = TXT_PRI)
                BasicTextField(
                    value = renameText, onValueChange = { renameText = it }, singleLine = true,
                    textStyle = androidx.compose.ui.text.TextStyle(color = TXT_PRI, fontSize = 13.sp),
                    keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                    keyboardActions = KeyboardActions(onDone = {
                        if (renameText.isNotBlank()) onRename(res, renameText); renamingResult = null
                    }),
                    modifier = Modifier.fillMaxWidth().background(BG_CHIP, RoundedCornerShape(8.dp))
                        .border(0.5.dp, ACCENT, RoundedCornerShape(8.dp)).padding(horizontal = 12.dp, vertical = 10.dp)
                )
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    OutlinedButton(onClick = { renamingResult = null }, modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = TXT_SEC),
                        border = androidx.compose.foundation.BorderStroke(0.5.dp, DIVIDER), shape = RoundedCornerShape(8.dp)
                    ) { Text("Cancel", fontSize = 12.sp) }
                    Button(onClick = { if (renameText.isNotBlank()) onRename(res, renameText); renamingResult = null },
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = ACCENT_DIM, contentColor = ACCENT),
                        shape = RoundedCornerShape(8.dp)
                    ) { Text("Save", fontSize = 12.sp) }
                }
            }
        }
    }

    if (results.isEmpty()) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { Text("No transcriptions yet", color = TXT_SEC) }
        return
    }
    LazyColumn(modifier = Modifier.fillMaxSize().padding(horizontal = 20.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp), contentPadding = PaddingValues(vertical = 20.dp)
    ) {
        item { Text("Transcriptions", fontSize = 11.sp, color = TXT_HINT) }
        items(results) { res ->
            ResultListItem(result = res, onSelect = { onSelect(res) },
                onRename = { renameText = res.name; renamingResult = res },
                onDelete = { onDelete(res) })
        }
    }
}

@Composable
fun ResultListItem(result: TranscriptionResult, onSelect: () -> Unit, onRename: () -> Unit, onDelete: () -> Unit) {
    var menuExpanded by remember { mutableStateOf(false) }
    Row(modifier = Modifier.fillMaxWidth().background(BG_CARD, RoundedCornerShape(10.dp))
        .border(0.5.dp, DIVIDER, RoundedCornerShape(10.dp))
        .clickable(interactionSource = remember { MutableInteractionSource() }, indication = null) { onSelect() }
        .padding(start = 14.dp, end = 4.dp, top = 12.dp, bottom = 12.dp),
        horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(result.name, fontSize = 13.sp, color = TXT_PRI)
            Spacer(Modifier.height(3.dp))
            Text("~${result.bpm} BPM  ·  %.1f KB".format(result.midiFile.length() / 1024f),
                fontSize = 10.sp, color = TXT_HINT, fontFamily = FontFamily.Monospace)
        }
        StatusChip(RecordingStatus.DONE)
        Spacer(Modifier.width(4.dp))
        Box {
            Icon(Icons.Default.MoreVert, contentDescription = "Options", tint = TXT_HINT,
                modifier = Modifier.size(20.dp).clickable(
                    interactionSource = remember { MutableInteractionSource() }, indication = null
                ) { menuExpanded = true })
            DropdownMenu(expanded = menuExpanded, onDismissRequest = { menuExpanded = false },
                modifier = Modifier.background(BG_CARD)
            ) {
                DropdownMenuItem(text = { Text("Rename", fontSize = 13.sp, color = TXT_PRI) },
                    onClick = { menuExpanded = false; onRename() })
                DropdownMenuItem(text = { Text("Delete", fontSize = 13.sp, color = ERROR_CLR) },
                    onClick = { menuExpanded = false; onDelete() })
            }
        }
    }
}

// Result detail screen
@Composable
fun ResultScreen(
    result: TranscriptionResult?,
    onExportMidi:   () -> Unit,
    onExportVideo:  () -> Unit,
    onNewRecording: () -> Unit
) {
    if (result == null) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { Text("No result yet", color = TXT_SEC) }
        return
    }
    val context = LocalContext.current
    var isPlaying by remember { mutableStateOf(false) }
    var videoError by remember { mutableStateOf(false) }
    var isPrepared by remember { mutableStateOf(false) }

    LazyColumn(modifier = Modifier.fillMaxSize().background(BG_DEEP).padding(horizontal = 20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp), contentPadding = PaddingValues(vertical = 20.dp)
    ) {
        item {
            Column(modifier = Modifier.fillMaxWidth().background(BG_CARD, RoundedCornerShape(12.dp))
                .border(0.5.dp, DIVIDER, RoundedCornerShape(12.dp))
            ) {
                Row(modifier = Modifier.fillMaxWidth().padding(horizontal = 14.dp, vertical = 8.dp),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text("Piano roll", fontSize = 11.sp, color = TXT_HINT, fontFamily = FontFamily.Monospace)
                    Text("~${result.bpm} BPM", fontSize = 11.sp, color = TXT_HINT, fontFamily = FontFamily.Monospace)
                }
                HorizontalDivider(color = DIVIDER, thickness = 0.5.dp)
                if (videoError || !result.videoFile.exists()) {
                    Box(
                        modifier = Modifier.fillMaxWidth().height(200.dp).background(BG_CARD),
                        contentAlignment = Alignment.Center
                    ) {
                        Text("Video preview unavailable", fontSize = 12.sp, color = TXT_SEC)
                    }
                } else {
                    AndroidView(
                        factory = { ctx ->
                            VideoView(ctx).apply {
                                setZOrderMediaOverlay(false)
                                setOnErrorListener { _, what, extra ->
                                    Log.e("VideoView", "Playback error: what=$what extra=$extra")
                                    videoError = true
                                    true
                                }
                                try {
                                    setVideoURI(Uri.fromFile(result.videoFile))
                                    setOnPreparedListener { mp ->
                                        try {
                                            mp.isLooping = true
                                            // Pre-warm the decoder so the first real
                                            // play is smooth: briefly start muted,
                                            // pause, and seek back to frame 0. This
                                            // forces MediaCodec to initialise and
                                            // decode the first frames now, instead of
                                            // stalling on the user's first Play tap.
                                            mp.setVolume(0f, 0f)
                                            mp.start()
                                            mp.pause()
                                            mp.seekTo(0)
                                            mp.setVolume(1f, 1f)
                                            isPrepared = true
                                            if (isPlaying) mp.start()
                                        } catch (e: Exception) {
                                            Log.e("VideoView", "onPrepared error", e)
                                            videoError = true
                                        }
                                    }
                                } catch (e: Exception) {
                                    Log.e("VideoView", "Could not load video", e)
                                    videoError = true
                                }
                            }
                        },
                        update = { videoView ->
                            try {
                                if (isPrepared) {
                                    if (isPlaying) videoView.start() else videoView.pause()
                                }
                            } catch (e: Exception) {
                                Log.e("VideoView", "Playback toggle failed", e)
                                videoError = true
                            }
                        },
                        modifier = Modifier.fillMaxWidth().height(200.dp)
                    )
                }
            }
        }
        item {
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinedButton(
                    onClick = {
                        if (videoError) {
                            Toast.makeText(context, "Video unavailable", Toast.LENGTH_SHORT).show()
                        } else {
                            isPlaying = !isPlaying
                        }
                    },
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = if (isPlaying) ACCENT else TXT_SEC),
                    border = androidx.compose.foundation.BorderStroke(0.5.dp, if (isPlaying) ACCENT else DIVIDER),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Icon(Icons.Default.PlayArrow, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(4.dp))
                    Text(if (isPlaying) "Pause" else "Play", fontSize = 12.sp)
                }
                Button(
                    onClick = onExportMidi, modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(containerColor = ACCENT_DIM, contentColor = ACCENT),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Icon(Icons.Default.KeyboardArrowDown, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(4.dp))
                    Text("Export MIDI", fontSize = 12.sp)
                }
            }
        }
        item {
            OutlinedButton(
                onClick = onExportVideo, modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.outlinedButtonColors(contentColor = TXT_SEC),
                border = androidx.compose.foundation.BorderStroke(0.5.dp, DIVIDER),
                shape = RoundedCornerShape(8.dp)
            ) {
                Icon(Icons.Default.KeyboardArrowDown, contentDescription = null, modifier = Modifier.size(16.dp))
                Spacer(Modifier.width(4.dp))
                Text("Export Video", fontSize = 12.sp)
            }
        }
        item {
            Column(modifier = Modifier.fillMaxWidth().background(BG_CARD, RoundedCornerShape(10.dp))
                .border(0.5.dp, DIVIDER, RoundedCornerShape(10.dp)).padding(14.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Text(result.midiFile.name, fontSize = 13.sp, color = TXT_PRI)
                    Text("%.1f KB".format(result.midiFile.length() / 1024f),
                        fontSize = 11.sp, color = TXT_HINT, fontFamily = FontFamily.Monospace)
                }
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    StatBox("tempo", "${result.bpm}", Modifier.weight(1f))
                    StatBox("midi",  "%.1f KB".format(result.midiFile.length() / 1024f), Modifier.weight(1f))
                    StatBox("video", "%.1f MB".format(result.videoFile.length() / 1_048_576f), Modifier.weight(1f))
                }
            }
        }
        item {
            OutlinedButton(onClick = onNewRecording, modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.outlinedButtonColors(contentColor = TXT_SEC),
                border = androidx.compose.foundation.BorderStroke(0.5.dp, DIVIDER), shape = RoundedCornerShape(10.dp)
            ) {
                Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(16.dp))
                Spacer(Modifier.width(6.dp))
                Text("New recording", fontSize = 13.sp)
            }
        }
    }
}

@Composable
fun StatBox(label: String, value: String, modifier: Modifier = Modifier) {
    Column(modifier = modifier.background(BG_CHIP, RoundedCornerShape(8.dp)).padding(10.dp)) {
        Text(label, fontSize = 9.sp, color = TXT_HINT)
        Spacer(Modifier.height(2.dp))
        Text(value, fontSize = 14.sp, fontWeight = FontWeight.Medium, color = TXT_PRI, fontFamily = FontFamily.Monospace)
    }
}

// Piano roll Canvas (kept for reference / offline use)
@Composable
fun PianoRollCanvas(notes: List<NoteEvent>, modifier: Modifier = Modifier) {
    val accentArgb = ACCENT.toArgb()
    val bgCardArgb = BG_CARD.toArgb()
    val bgChipArgb = BG_CHIP.toArgb()
    Canvas(modifier = modifier) {
        val W = size.width; val H = size.height
        if (notes.isEmpty()) return@Canvas
        val maxPitch   = notes.maxOf { it.pitch }
        val pitchRange = (maxPitch - notes.minOf { it.pitch } + 1).coerceAtLeast(12)
        val totalBeats = notes.maxOf { it.startBeat + it.durationBeats }.coerceAtLeast(8f)
        val noteH      = H / pitchRange
        drawIntoCanvas { cc ->
            val nc = cc.nativeCanvas
            val rowPaint  = Paint().apply { isAntiAlias = false }
            val linePaint = Paint().apply { strokeWidth = 0.5f }
            val notePaint = Paint().apply { isAntiAlias = true }
            for (i in 0 until pitchRange) {
                rowPaint.color = if (i % 2 == 0) bgChipArgb else bgCardArgb
                nc.drawRect(0f, i * noteH, W, (i + 1) * noteH, rowPaint)
            }
            for (b in 0..totalBeats.toInt()) {
                val x = (b / totalBeats) * W
                linePaint.color = if (b % 4 == 0) Color.argb(80,255,255,255) else Color.argb(30,255,255,255)
                nc.drawLine(x, 0f, x, H, linePaint)
            }
            notes.forEach { note ->
                val x = (note.startBeat / totalBeats) * W + 1f
                val w = ((note.durationBeats / totalBeats) * W - 2f).coerceAtLeast(4f)
                val y = (maxPitch - note.pitch) * noteH + 1f
                val alpha = (0.65f + (note.pitch % 5) * 0.07f).coerceIn(0f, 1f)
                notePaint.color = Color.argb((alpha * 255).toInt(),
                    (accentArgb shr 16) and 0xFF, (accentArgb shr 8) and 0xFF, accentArgb and 0xFF)
                nc.drawRoundRect(RectF(x, y, x + w, y + noteH - 2f), 2f, 2f, notePaint)
            }
        }
    }
}