package com.marusys.hesap

import android.Manifest
import android.app.ActivityManager
import android.app.AlertDialog
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import com.marusys.hesap.feature.MemoryUsageManager
import com.marusys.hesap.presentation.screen.AudioScreen
import com.marusys.hesap.presentation.viewmodel.MainViewModel
import com.marusys.hesap.service.AudioService
import kotlin.math.exp

class MainActivity : ComponentActivity() {
    // 여러 페이지에서 사용하는 값을 관리하는 viewModel
    private val mainViewModel = MainViewModel()
    private val OVERLAY_PERMISSION_REQUEST_CODE = 1
    private val handler = Handler(Looper.getMainLooper())
    private lateinit var memoryUsageManager: MemoryUsageManager

    // AudioService에서 방송하는걸 MainActivity에서 받아서 객체? (정확한 명칭 모름) 로 정의
    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action == "SPEECH_RECOGNITION_RESULT") {
                val matches = intent.getStringArrayListExtra("matches")
                mainViewModel.setResultText(matches?.firstOrNull() ?: "")
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // 오디오 녹음 권한이 있는지 확인
//        if (ContextCompat.checkSelfPermission(
//                this,
//                Manifest.permission.RECORD_AUDIO
//            ) != PackageManager.PERMISSION_GRANTED
//        ) {
//            // 권한이 없으면 요청 메서드 실행
//            ActivityCompat.requestPermissions(
//                this,
//                arrayOf(Manifest.permission.RECORD_AUDIO),
//                1
//            )
//        }else{
//            realTimeRecordAndClassify()
//        }

        // 권한 체크 및 요청
        checkAndRequestPermissions()

        // 오버레이 권한 체크 및 요청
        checkAndRequestOverlayPermission()

        // AudioService 시작
        startAudioService()

        // BroadcastReceiver 등록
        registerBroadcastReceiver()
        // 메모리 사용량 관리자 초기화
        initializeMemoryUsageManager()

        // UI 설정
        setContent {
            AudioScreen(
                viewModel = mainViewModel,
                recordButtons = { recordAndClassify() },
                startService = { startAudioService() },
            )
        }
    }

    private fun checkAndRequestPermissions() {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED ||
            checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(
                arrayOf(Manifest.permission.RECORD_AUDIO, Manifest.permission.CAMERA),
                1
            )
        }
    }

    private fun checkAndRequestOverlayPermission() {
        if (!Settings.canDrawOverlays(this)) {
            val intent =
                Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:$packageName"))
            startActivityForResult(intent, OVERLAY_PERMISSION_REQUEST_CODE)
        }
    }

    private fun startAudioService() {
        //
        if (!isServiceRunning(AudioService::class.java)) {
            val serviceIntent = Intent(this, AudioService::class.java)
            ContextCompat.startForegroundService(this, serviceIntent)
        }
        //
//        val serviceIntent = Intent(this, AudioService::class.java)
        // 포그라운드 서비스 시작
//        ContextCompat.startForegroundService(this, serviceIntent)
    }

    private fun isServiceRunning(serviceClass: Class<*>): Boolean {
        val manager = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        return manager.getRunningServices(Integer.MAX_VALUE)
            .any { it.service.className == serviceClass.name }
    }

    private fun registerBroadcastReceiver() {
        LocalBroadcastManager.getInstance(this).registerReceiver(
            receiver,
            IntentFilter("SPEECH_RECOGNITION_RESULT")
        )
    }

    private val updateMemoryRunnable = object : Runnable {
        override fun run() {
            mainViewModel.setMemoryText(memoryUsageManager.getMemoryUsage())
            handler.postDelayed(this, 1000) // 1초마다 업데이트
        }
    }

    private fun initializeMemoryUsageManager() {
        memoryUsageManager = MemoryUsageManager(this)
    }

    override fun onResume() {
        super.onResume()
        handler.post(updateMemoryRunnable)
    }

    override fun onPause() {
        super.onPause()
        handler.removeCallbacks(updateMemoryRunnable)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == OVERLAY_PERMISSION_REQUEST_CODE) {
            if (Settings.canDrawOverlays(this)) {
                startService(Intent(this, AudioService::class.java))
            } else {
                Toast.makeText(this, "오버레이 권한이 필요합니다", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        LocalBroadcastManager.getInstance(this).unregisterReceiver(receiver)
    }

    // ======== 음성 인식 기반 분류 ========
    // 분류할 라벨들 -> 모델 학습 시 사용한 라벨
    private val labels = arrayOf(
        "unknown",
        "ssafy"
    )

    // 호출어 인식 여부에 따라 스레드 일시 중단 시키기 위한 변수
    private var isListening = false

    fun realTimeRecordAndClassify() {
        val sampleRate = 16000
        val windowSize = 16000  // 1초 분량의 샘플 (16000개)
        val stepSize = 8000     // 0.5초 분량의 샘플 (겹치는 구간)

        val bufferSize = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        if (ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            return
        }

        // 상태 표시
        runOnUiThread {
            mainViewModel.setResultText("녹음 중...")
        }

        Thread {
            isListening = true  // 스레드 실행 시작
            val audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize
            )
            if (audioRecord.state != AudioRecord.STATE_INITIALIZED) {
                Log.e("MainActivity", "AudioRecord 초기화 실패")
                runOnUiThread {
                    mainViewModel.setResultText("녹음 초기화 실패")
                }
                return@Thread
            }

            val audioBuffer = ShortArray(bufferSize / 2)
            val slidingWindowBuffer = FloatArray(windowSize)  // 1초 버퍼
            var bufferPosition = 0

            audioRecord.startRecording()

            // 실시간으로 데이터를 읽어들여 모델로 전달
            while (isListening) {
                val readSize = audioRecord.read(audioBuffer, 0, audioBuffer.size)
                if (readSize > 0) {
                    for (i in 0 until readSize) {
                        slidingWindowBuffer[bufferPosition] = audioBuffer[i] / 32768.0f
                        bufferPosition++

                        // 슬라이딩 윈도우가 채워졌으면 호출어 검출을 수행
                        if (bufferPosition >= windowSize) {
                            bufferPosition = 0

                            try {
                                val classifier = AudioClassifier(this)
                                val inputBuffer = classifier.createInputBuffer(slidingWindowBuffer)
                                val results = classifier.classify(inputBuffer)

                                // results[0] 값을 실시간으로 화면에 표시
                                runOnUiThread {
                                    mainViewModel.setResultText("확률값: ${results[0]}")
                                }

                                // 호출어가 감지되면 팝업을 띄우고 스레드를 중단
                                if (results[0] >= 0.9f) {
                                    runOnUiThread {
                                        showSuccessDialog()
                                    }
                                    isListening = false  // 스레드 중단
                                    break  // 루프 종료
                                }
                            } catch (e: Exception) {
                                Log.e("MainActivity", "분류 중 오류 발생", e)
                                runOnUiThread {
                                    mainViewModel.setResultText("분류 중 오류가 발생했습니다: " + e.message)
                                }
                            }

                            // 슬라이딩 윈도우를 50% 이동시키기 위해 이전 데이터를 복사
                            System.arraycopy(
                                slidingWindowBuffer,
                                stepSize,
                                slidingWindowBuffer,
                                0,
                                windowSize - stepSize
                            )
                            bufferPosition = windowSize - stepSize
                        }
                    }
                }
            }
            audioRecord.stop()
            audioRecord.release()
        }.start()
    }

    // 호출어 인식 성공 시 보여줄 팝업
    private fun showSuccessDialog() {
        AlertDialog.Builder(this)
            .setTitle("호출어 인식 성공")
            .setMessage("호출어가 성공적으로 인식되었습니다!")
            .setPositiveButton("확인") { dialog, _ ->
                dialog.dismiss()  // 팝업 닫기
                realTimeRecordAndClassify()  // 스레드 재시작
            }
            .setCancelable(false)
            .show()
    }

    // 현재는 버튼 리스너 기반 -> 추후에 실시간 음성인식 코드 구현
    fun recordAndClassify() {
        // 샘플 레이트 16KHz(16000Hz)
        val sampleRate = 16000
        // 녹음 설정 시간 ( 1초로 설정 )
        val recordingTime = 1
        // 샘플 갯수 계산
        val totalSamples = sampleRate * recordingTime
        // 최소 버퍼 크기를 얻어옴
        val bufferSize = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        // 녹음 권한이 있는지 재확인
        if (ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            return
        }

        // 녹음 상태 표시해주기 (스레드 별도 설정)
        runOnUiThread {
            mainViewModel.setResultText("녹음 중...")
        }

        // 백그라운드 스레드에서 녹음 및 분류 실행
        Thread(Runnable {
            val audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,  // 마이크에서 오디오 소스 가져옴
                sampleRate,  // 샘플레이트 설정 (16kHz)
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize // 버퍼 크기 설정
            )
            // AudioRecord 초기화 실패 시 로그 출력 및 종료
            if (audioRecord.state != AudioRecord.STATE_INITIALIZED) {
                Log.e("MainActivity", "AudioRecord 초기화 실패")
                runOnUiThread {
                    mainViewModel.setResultText("녹음 초기화 실패")
                }
                return@Runnable
            }

            // 녹음할 샘플을 저장할 버퍼 생성
            val audioBuffer = ShortArray(totalSamples)

            // 녹음 시작
            audioRecord.startRecording()

            // 녹음 데이터 읽기
            audioRecord.read(audioBuffer, 0, audioBuffer.size)

            // 녹음 종료 & 리소스 해제
            audioRecord.stop()
            audioRecord.release()

            Log.d("원시 데이터", audioBuffer.contentToString())

            // short 배열을 float 배열로 변환 (정규화 포함)
            val audioData = FloatArray(16000)
            for (i in audioData.indices) {
                // 16비트 데이터 정규화 (-1.0 ~ 1.0 값으로 맞춰줌)
                audioData[i] = audioBuffer[i] / 32768.0f
            }

            // 입력 음성 데이터 값 로그
            Log.d("audioData", audioData.contentToString())
            try {
                // 자체 AudioClassifier를 사용하여 분류
                val classifier = AudioClassifier(this)

                // 입력 데이터를 분류 모델의 형식에 맞게 변환
                val inputBuffer = classifier.createInputBuffer(audioData)
                val results = classifier.classify(inputBuffer)

                // 소프트맥스 적용
                var max = Float.NEGATIVE_INFINITY
                for (result in results) {
                    if (result > max) max = result
                }

                var sum = 0f
                val softmaxResults = FloatArray(results.size)

                for (i in results.indices) {
                    softmaxResults[i] = exp((results[i] - max).toDouble()).toFloat()
                    sum += softmaxResults[i]
                }

                // 소프트맥스 결과 정규화
                for (i in softmaxResults.indices) {
                    softmaxResults[i] /= sum
                }

                // 결과 문자열 생성 (결과 값 포맷팅)
                val resultText = StringBuilder()
                for (i in results.indices) {
                    resultText.append(labels[i])
                        .append(" : ")
                        .append(String.format("%.4f", results[i]))
                        .append("(")
                        .append(String.format("%.2f", softmaxResults[i] * 100))
                        .append("%)\n")
                }

                val finalResult = resultText.toString()

                // 결과 출력
                runOnUiThread {
                    mainViewModel.setResultText(finalResult)
                }
                Log.d("resultText", finalResult)
            } catch (e: Exception) {
                Log.e("MainActivity", "분류 중 오류 발생", e)
                runOnUiThread {
                    mainViewModel.setResultText("분류 중 오류가 발생했습니다: " + e.message)
                }
            }
        }).start()
    } // ======== wav 파일 기반 분류 ========
}



