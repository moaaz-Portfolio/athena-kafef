package com.moaaz.radaralerts

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.ViewGroup
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.io.IOException
import java.util.concurrent.Executors

class MainActivity : Activity() {
    private val executor = Executors.newSingleThreadExecutor()
    private lateinit var serverUrl: EditText
    private lateinit var apiTokenInput: EditText
    private lateinit var buzzer: CheckBox
    private val switches = linkedMapOf<String, CheckBox>()

    private val alertOptions = linkedMapOf(
        "fall" to "اشتباه سقوط",
        "movement" to "اكتشاف الحركة",
        "presence" to "وجود شخص",
        "empty_room" to "الغرفة أصبحت فارغة",
        "door_presence" to "شخص عند الباب",
        "device_offline" to "انقطاع اتصال الجهاز",
        "csi_quality" to "انخفاض جودة إشارة CSI"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val prefs = getSharedPreferences("radar", 0)
        val pad = (18 * resources.displayMetrics.density).toInt()
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
            setBackgroundColor(0xFFF1F5F9.toInt())
        }
        val scroll = ScrollView(this).apply { addView(root) }

        root.addView(TextView(this).apply {
            text = "WiFi Radar"
            textSize = 27f
            setTextColor(0xFF14213D.toInt())
        })
        root.addView(TextView(this).apply {
            text = "إعداد اتصال التطبيق والتنبيهات المشتركة مع لوحة الويب"
            textSize = 14f
        })

        root.addView(TextView(this).apply {
            text = "عنوان جهاز المعالجة داخل الشبكة المحلية"
            textSize = 14f
        })
        serverUrl = EditText(this).apply {
            hint = "http://192.168.1.100:8000"
            setSingleLine(true)
            setText(prefs.getString("server_url", "http://192.168.1.100:8000"))
            inputType = android.text.InputType.TYPE_TEXT_VARIATION_URI
        }
        root.addView(serverUrl, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)

        root.addView(TextView(this).apply {
            text = "رمز الربط المعروض في لوحة الويب على جهاز المعالجة"
            textSize = 14f
        })
        apiTokenInput = EditText(this).apply {
            hint = "ألصق رمز الربط هنا"
            setSingleLine(true)
            setText(prefs.getString("api_token", ""))
            inputType = android.text.InputType.TYPE_CLASS_TEXT
        }
        root.addView(apiTokenInput, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)

        val sync = Button(this).apply { text = "مزامنة الإعدادات من الجهاز" }
        root.addView(sync)

        root.addView(TextView(this).apply {
            text = "أنواع الإشعارات"
            textSize = 19f
            setTextColor(0xFF14213D.toInt())
        })
        alertOptions.forEach { (key, label) ->
            val check = CheckBox(this).apply {
                text = label
                isChecked = true
            }
            switches[key] = check
            root.addView(check)
        }

        buzzer = CheckBox(this).apply {
            text = "السماح بتنبيه Buzzer عند ربط وحدة الباب"
            isChecked = false
        }
        root.addView(buzzer)

        val save = Button(this).apply { text = "حفظ ومزامنة الإعدادات" }
        val start = Button(this).apply { text = "تشغيل إشعارات WiFi Radar" }
        val stop = Button(this).apply { text = "إيقاف مراقبة التطبيق" }
        root.addView(save)
        root.addView(start)
        root.addView(stop)

        root.addView(TextView(this).apply {
            text = "يجب أن يكون الهاتف وجهاز المعالجة على الشبكة المحلية نفسها. شغّل User Edition على جهاز المعالجة مع --host 0.0.0.0. الإشعارات عن بُعد خارج الشبكة تحتاج اتصالًا وخدمة دفع مناسبة."
            textSize = 12f
        })

        setContentView(scroll)

        sync.setOnClickListener { syncFromServer() }
        save.setOnClickListener { saveToServer() }

        start.setOnClickListener {
            val base = normalizedUrl()
            if (base == null) {
                Toast.makeText(this, "اكتب عنوانًا صحيحًا يبدأ بـ http:// أو https://", Toast.LENGTH_LONG).show()
                return@setOnClickListener
            }
            prefs.edit().putString("server_url", base).apply()
            if (Build.VERSION.SDK_INT >= 33 &&
                checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
            ) {
                requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 10)
            }
            val intent = Intent(this, RadarService::class.java)
            if (Build.VERSION.SDK_INT >= 26) startForegroundService(intent) else startService(intent)
            Toast.makeText(this, "بدأت مراقبة WiFi Radar", Toast.LENGTH_SHORT).show()
        }

        stop.setOnClickListener {
            stopService(Intent(this, RadarService::class.java))
            Toast.makeText(this, "تم إيقاف مراقبة التطبيق", Toast.LENGTH_SHORT).show()
        }
    }

    private fun normalizedUrl(): String? {
        val value = serverUrl.text.toString().trim().trimEnd('/')
        return if (value.startsWith("http://") || value.startsWith("https://")) value else null
    }

    private fun syncFromServer() {
        val base = normalizedUrl() ?: run {
            Toast.makeText(this, "راجع عنوان الجهاز", Toast.LENGTH_SHORT).show()
            return
        }
        getSharedPreferences("radar", 0).edit().putString("server_url", base).apply()
        executor.execute {
            try {
                val result = request("$base/api/config")
                val notifications = result.optJSONObject("notifications") ?: JSONObject()
                val outputs = result.optJSONObject("outputs") ?: JSONObject()
                runOnUiThread {
                    switches.forEach { (key, check) ->
                        check.isChecked = notifications.optBoolean(key, true)
                    }
                    buzzer.isChecked = outputs.optBoolean("buzzer_enabled", false)
                    Toast.makeText(this, "تمت المزامنة مع ${result.optString("edition", "الجهاز")}", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                runOnUiThread {
                    Toast.makeText(this, "تعذر الاتصال: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun saveToServer() {
        val base = normalizedUrl() ?: run {
            Toast.makeText(this, "راجع عنوان الجهاز", Toast.LENGTH_SHORT).show()
            return
        }
        val token = apiTokenInput.text.toString().trim()
        if (token.isBlank()) {
            Toast.makeText(this, "ألصق رمز الربط من لوحة الويب أولًا", Toast.LENGTH_LONG).show()
            return
        }
        getSharedPreferences("radar", 0).edit()
            .putString("server_url", base)
            .putString("api_token", token)
            .apply()
        val notifications = JSONObject()
        switches.forEach { (key, check) -> notifications.put(key, check.isChecked) }
        val outputs = JSONObject().put("buzzer_enabled", buzzer.isChecked)
        val payload = JSONObject()
            .put("notifications", notifications)
            .put("outputs", outputs)

        executor.execute {
            try {
                request("$base/api/config", "POST", payload)
                runOnUiThread {
                    Toast.makeText(this, "تم حفظ الإعدادات على الجهاز ومزامنتها مع الويب", Toast.LENGTH_LONG).show()
                }
            } catch (e: Exception) {
                runOnUiThread {
                    Toast.makeText(this, "فشل الحفظ: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    @Throws(IOException::class)
    private fun request(address: String, method: String = "GET", body: JSONObject? = null): JSONObject {
        val connection = URL(address).openConnection() as HttpURLConnection
        try {
            connection.connectTimeout = 3500
            connection.readTimeout = 3500
            connection.requestMethod = method
            connection.setRequestProperty("Accept", "application/json")
            if (body != null) {
                connection.doOutput = true
                val token = getSharedPreferences("radar", 0).getString("api_token", "") ?: ""
                connection.setRequestProperty("Authorization", "Bearer $token")
                connection.setRequestProperty("Content-Type", "application/json; charset=utf-8")
                connection.outputStream.use { it.write(body.toString().toByteArray(Charsets.UTF_8)) }
            }
            val code = connection.responseCode
            val stream = if (code in 200..299) connection.inputStream else connection.errorStream
            val content = stream?.bufferedReader()?.use { it.readText() } ?: "{}"
            if (code !in 200..299) throw IOException("HTTP $code: $content")
            return JSONObject(content)
        } finally {
            connection.disconnect()
        }
    }

    override fun onDestroy() {
        executor.shutdownNow()
        super.onDestroy()
    }
}
