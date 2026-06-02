package es.uc3m.android.pianotranscriber

import android.content.Context
import android.media.MediaRecorder
import android.os.Build
import android.util.Log
import java.io.File
import java.io.IOException

/**
 * Thin wrapper around [MediaRecorder] for capturing audio to a file.
 *
 * Usage:
 *   val recorder = AudioRecorder(context, file)
 *   val ok = recorder.start()
 *   // … later …
 *   val seconds = recorder.stop()
 */
class AudioRecorder(
    private val context: Context,
    private val outputFile: File
) {
    private var recorder: MediaRecorder? = null
    private var isRecording = false
    private var startTimeMs: Long = 0L

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Starts recording.
     * @return true if recording started successfully, false otherwise.
     *         Callers should not flip UI state to "recording" unless this returns true.
     */
    fun start(): Boolean {
        if (isRecording) return true

        // Ensure the parent directory exists; MediaRecorder will not create it.
        try {
            outputFile.parentFile?.mkdirs()
        } catch (e: SecurityException) {
            Log.e(TAG, "Cannot access output directory", e)
            return false
        }

        recorder = try {
            buildRecorder()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to instantiate MediaRecorder", e)
            return false
        }

        // Listeners must be set BEFORE prepare(); otherwise native events are dropped.
        recorder!!.setOnErrorListener { _, what, extra ->
            Log.e(TAG, "MediaRecorder error: what=$what extra=$extra")
            stop()
        }
        recorder!!.setOnInfoListener { _, what, extra ->
            Log.d(TAG, "MediaRecorder info: what=$what extra=$extra")
        }

        try {
            recorder!!.apply {
                setAudioSource(MediaRecorder.AudioSource.MIC)
                setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
                setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
                setAudioSamplingRate(44_100)
                setAudioEncodingBitRate(128_000)
                setOutputFile(outputFile.absolutePath)
            }
        } catch (e: IllegalStateException) {
            Log.e(TAG, "MediaRecorder configuration failed (illegal state)", e)
            safeReleaseRecorder()
            return false
        } catch (e: Exception) {
            Log.e(TAG, "MediaRecorder configuration failed", e)
            safeReleaseRecorder()
            return false
        }

        return try {
            recorder!!.prepare()
            recorder!!.start()
            isRecording = true
            startTimeMs = System.currentTimeMillis()
            Log.d(TAG, "Recording started → ${outputFile.absolutePath}")
            true
        } catch (e: IOException) {
            // prepare() throws IOException if the output file is unwritable
            Log.e(TAG, "Failed to prepare recorder (I/O)", e)
            safeReleaseRecorder()
            false
        } catch (e: IllegalStateException) {
            // start() throws IllegalStateException if the mic is already in use
            // or the recorder is in an invalid state
            Log.e(TAG, "Failed to start recorder (illegal state — mic busy?)", e)
            safeReleaseRecorder()
            false
        } catch (e: SecurityException) {
            // Missing RECORD_AUDIO permission
            Log.e(TAG, "Failed to start recorder (permission denied)", e)
            safeReleaseRecorder()
            false
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start recording", e)
            safeReleaseRecorder()
            false
        }
    }

    /**
     * Stops recording and releases all resources.
     * @return duration in seconds, or 0 if not recording or if stop failed.
     */
    fun stop(): Int {
        if (!isRecording) return 0

        val durationSec = ((System.currentTimeMillis() - startTimeMs) / 1000).toInt()
        var stopFailed = false

        try {
            recorder?.stop()
        } catch (e: IllegalStateException) {
            // stop() throws if called too soon after start() — file will be invalid
            Log.e(TAG, "Error stopping recorder (called too soon?)", e)
            stopFailed = true
        } catch (e: RuntimeException) {
            // MediaRecorder.stop() officially documents this for too-short recordings
            Log.e(TAG, "Error stopping recorder (runtime)", e)
            stopFailed = true
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping recorder", e)
            stopFailed = true
        } finally {
            // reset() before release() is required in all exit paths
            safeReleaseRecorder()
            isRecording = false
        }

        // If stop() failed, the file is likely corrupt or zero-bytes —
        // delete it so callers don't try to upload garbage.
        if (stopFailed) {
            try {
                if (outputFile.exists() && outputFile.length() < 1024L) {
                    outputFile.delete()
                    Log.w(TAG, "Deleted corrupt recording file")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to delete corrupt file", e)
            }
            return 0
        }

        Log.d(TAG, "Recording stopped — duration=${durationSec}s → ${outputFile.absolutePath}")
        return durationSec
    }

    fun isActive(): Boolean = isRecording

    fun outputPath(): String = outputFile.absolutePath

    // ── Private helpers ───────────────────────────────────────────────────────

    private fun safeReleaseRecorder() {
        try {
            recorder?.reset()
        } catch (e: Exception) {
            Log.e(TAG, "Error resetting recorder", e)
        }
        try {
            recorder?.release()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing recorder", e)
        }
        recorder = null
    }

    private fun buildRecorder(): MediaRecorder =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            MediaRecorder(context)
        } else {
            @Suppress("DEPRECATION")
            MediaRecorder()
        }

    companion object {
        private const val TAG = "AudioRecorder"
    }
}