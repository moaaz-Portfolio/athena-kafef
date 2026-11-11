package com.moaaz.radaralerts

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.Build
import android.os.IBinder
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.TimeUnit

class RadarService : Service() {
    private val statusChannel = "wifi_radar_status"
    private val alertsChannel = "wifi_radar_alerts"
    private val foregroundId = 100
    private var worker: ScheduledExecutorService? = null

    private var wasConnected: Boolean? = null
    private var wasMoving: Boolean? = null
    private var previousPresence: String? = null
    private var wasFallAlert = false
    private var wasLowCsi = false
    private var wasDoorPresent: Boolean? = null

    override fun onCreate() {
        super.onCreate()
        val manager = getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= 26) {
            manager.createNotificationChannel(
                NotificationChannel(statusChannel, "حالة WiFi Radar", NotificationManager.IMPORTANCE_LOW)
            )
            manager.createNotificationChannel(
                NotificationChannel(alertsChannel, "تنبيهات WiFi Radar", NotificationManager.IMPORTANCE_HIGH)
            )
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForeground(foregroundId, notification(statusChannel, "المراقبة المحلية تعمل", "WiFi Radar"))
        if (worker == null || worker?.isShutdown == true) {
            worker = Executors.newSingleThreadScheduledExecutor()
            worker?.scheduleWithFixedDelay({ poll() }, 0, 3, TimeUnit.SECONDS)
        }
        return START_STICKY
    }

    private fun poll() {
        val base = getSharedPreferences("radar", 0)
            .getString("server_url", "http://192.168.1.100:8000")
            ?.trimEnd('/') ?: return

        try {
            val state = getJson("$base/api/state")
            val config = getJson("$base/api/config")
            val notifications = config.optJSONObject("notifications") ?: JSONObject()

            val connected = state.optBoolean("connected", false)
            val monitor = state.optJSONObject("monitor") ?: JSONObject()
            val fall = state.optJSONObject("fall") ?: JSONObject()
            val moving = monitor.optString("motion", "unknown") == "moving"
            val presence = monitor.optString("status", "unknown")
            val fallAlert = fall.optString("status", "ready") == "alert"
            val lowCsi = state.optDouble("loss", 0.0) >= 10.0
            val door = state.optJSONObject("door") ?: JSONObject()
            val doorPresent = door.optString("status", "unknown") == "present"

            if (wasConnected == true && !connected && notifications.optBoolean("device_offline", true)) {
                alert(201, "انقطع اتصال الجهاز", "تحقق من اتصال ESP32-S3 وجهاز المعالجة")
            }
            if (wasMoving != true && moving && notifications.optBoolean("movement", true)) {
                alert(202, "تم اكتشاف حركة", "WiFi Radar رصد حركة داخل الغرفة")
            }
            if (previousPresence != "present" && presence == "present" &&
                notifications.optBoolean("presence", true)
            ) {
                alert(203, "تم رصد وجود شخص", "يوجد شخص محتمل داخل الغرفة")
            }
            if (previousPresence != "empty" && presence == "empty" &&
                notifications.optBoolean("empty_room", false)
            ) {
                alert(204, "الغرفة تبدو فارغة", "لم يعد النظام يرصد وجودًا داخل الغرفة")
            }
            if (!wasFallAlert && fallAlert && notifications.optBoolean("fall", true)) {
                alert(205, "اشتباه سقوط", "تحقق من الشخص مباشرة؛ هذا تنبيه بحثي وليس تأكيدًا طبيًا")
            }
            if (!wasLowCsi && lowCsi && notifications.optBoolean("csi_quality", false)) {
                alert(206, "انخفاض جودة إشارة CSI", "فقد الحزم تجاوز 10%")
            }
            if (wasDoorPresent != true && doorPresent &&
                notifications.optBoolean("door_presence", true)
            ) {
                alert(207, "شخص عند الباب", "تم رصد وجود شخص أمام حساس الباب")
            }

            wasConnected = connected
            wasMoving = moving
            previousPresence = presence
            wasFallAlert = fallAlert
            wasLowCsi = lowCsi
            wasDoorPresent = doorPresent
        } catch (_: Exception) {
            if (wasConnected == true) {
                val config = runCatching { getJson("$base/api/config") }.getOrNull()
                val enabled = config?.optJSONObject("notifications")?.optBoolean("device_offline", true) ?: true
                if (enabled) alert(201, "تعذر الوصول إلى الجهاز", "افحص الشبكة المحلية وعنوان جهاز WiFi Radar")
            }
            wasConnected = false
        }
    }

    private fun getJson(address: String): JSONObject {
        val connection = URL(address).openConnection() as HttpURLConnection
        try {
            connection.connectTimeout = 2500
            connection.readTimeout = 2500
            connection.requestMethod = "GET"
            connection.setRequestProperty("Accept", "application/json")
            if (connection.responseCode != 200) throw Exception("HTTP")
            val content = connection.inputStream.bufferedReader().use { it.readText() }
            return JSONObject(content)
        } finally {
            connection.disconnect()
        }
    }

    private fun alert(id: Int, title: String, body: String) {
        getSystemService(NotificationManager::class.java)
            .notify(id, notification(alertsChannel, title, body))
    }

    private fun notification(channel: String, title: String, body: String): Notification {
        return Notification.Builder(this, channel)
            .setContentTitle(title)
            .setContentText(body)
            .setStyle(Notification.BigTextStyle().bigText(body))
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setOngoing(channel == statusChannel)
            .setAutoCancel(channel != statusChannel)
            .build()
    }

    override fun onDestroy() {
        worker?.shutdownNow()
        worker = null
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
