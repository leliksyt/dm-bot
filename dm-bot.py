import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import google.generativeai as genai
from instagrapi import Client


logger = logging.getLogger("dm_bot")


class StopRun(Exception):
    """Güvenli biçimde durdurulması gereken bir çalıştırma hatası."""


class UncertainSendError(StopRun):
    """Gönderimin gerçekleşip gerçekleşmediği doğrulanamadığında kullanılır."""


def configure_logging(log_file):
    """İşlem kayıtlarını belirtilen dosyaya yazar."""
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)


def configure_gemini(api_key):
    """Gemini API istemcisini yapılandırır ve modeli döndürür."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=genai.GenerationConfig(
            max_output_tokens=300,
            temperature=0.8,
        ),
    )
    logger.info("Gemini modeli yapılandırıldı.")
    return model


def generate_ai_reply(model, incoming_message, system_prompt):
    """
    Gemini ile gelen mesaja özel bir yanıt üretir.
    Hata durumunda yedek şablon metni döndürür.
    """
    prompt = f"{system_prompt}\n\nKullanıcı mesajı: {incoming_message}\n\nYanıt:"

    try:
        response = model.generate_content(prompt)
        reply = response.text.strip()
        logger.info("Gemini yanıt üretti.")
        return reply
    except Exception as error:
        logger.warning(
            "Gemini yanıt üretemedi (%s); yedek şablon kullanılıyor.",
            type(error).__name__,
        )
        return random.choice(FALLBACK_TEMPLATES)


def is_rate_limit_error(error):
    """Hatanın hız sınırı veya geçici kısıtlama hatası olup olmadığını belirler."""
    error_name = type(error).__name__.lower()
    error_text = str(error).lower()
    combined = f"{error_name} {error_text}"

    indicators = (
        "rate limit",
        "ratelimit",
        "too many requests",
        "please wait",
        "429",
        "throttl",
    )
    return any(indicator in combined for indicator in indicators)


def is_connection_error(error):
    """Hatanın bağlantı veya zaman aşımı kaynaklı olup olmadığını belirler."""
    error_name = type(error).__name__.lower()
    error_text = str(error).lower()
    combined = f"{error_name} {error_text}"

    indicators = (
        "connection",
        "network",
        "timeout",
        "timed out",
        "temporarily unavailable",
    )
    return any(indicator in combined for indicator in indicators)


def backoff_wait():
    """Yeniden denemeden önce birkaç dakika bekler ve beklemeyi kaydeder."""
    seconds = random.uniform(180, 300)
    logger.warning("Geçici sorun nedeniyle %.0f saniye bekleniyor.", seconds)
    time.sleep(seconds)


def login_client(username, password, session_file):
    """Kaydedilmiş oturumu yükleyerek giriş yapar ve güncel oturumu kaydeder."""
    client = Client()
    session_path = Path(session_file)

    if session_path.exists():
        try:
            client.load_settings(str(session_path))
            logger.info("Kaydedilmiş oturum ayarları yüklendi.")
        except Exception as error:
            logger.warning(
                "Oturum dosyası yüklenemedi (%s); yeni giriş denenecek.",
                type(error).__name__,
            )

    try:
        client.login(username, password)
    except Exception as error:
        logger.error("Giriş başarısız (%s).", type(error).__name__)
        raise StopRun("Instagram giriş işlemi başarısız.") from None

    client.dump_settings(str(session_path))
    logger.info("Giriş başarılı; oturum dosyası güncellendi.")
    return client


def load_sent_conversations(sent_file):
    """Daha önce yanıtlanan konuşma kimliklerini dosyadan okur."""
    path = Path(sent_file)

    if not path.exists():
        return set()

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        conversation_ids = data
    else:
        conversation_ids = data.get("sent_conversations", [])

    return {str(conversation_id) for conversation_id in conversation_ids}


def save_sent_conversations(sent_file, conversation_ids):
    """Yanıtlanan konuşma kimliklerini dosyaya atomik olarak kaydeder."""
    path = Path(sent_file)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "sent_conversations": sorted(
            {str(conversation_id) for conversation_id in conversation_ids}
        )
    }

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    os.replace(temporary_path, path)


def fetch_unread_threads(client):
    """Okunmamış DM konuşmalarını alır; geçici hatalarda sınırlı sayıda dener."""
    for attempt in range(2):
        try:
            threads = client.direct_threads(
                amount=100,
                selected_filter="unread",
                thread_message_limit=1,
            )
            logger.info("%d okunmamış konuşma bulundu.", len(threads))
            return threads
        except Exception as error:
            retryable = is_rate_limit_error(error) or is_connection_error(error)
            logger.error(
                "Okunmamış konuşmalar alınamadı (%s).",
                type(error).__name__,
            )

            if not retryable or attempt == 1:
                raise StopRun("Okunmamış konuşmalar alınamadı.") from None

            backoff_wait()

    return []


def get_last_incoming_message(thread):
    """
    Konuşmadaki son gelen mesajın metnini döndürür.
    Mesaj yoksa veya okunamıyorsa boş string döndürür.
    """
    messages = getattr(thread, "messages", None) or []
    for message in messages:
        text = getattr(message, "text", None)
        if text:
            return text.strip()
    return ""


def to_epoch(value):
    """Tarih/zaman değerini Unix zamanına çevirir; tanınmayan değerlerde None döndürür."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp()

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
        except ValueError:
            return None

    return None


def verify_reply_visible(client, thread_id, reply_text, attempted_at):
    """Bağlantı hatasından sonra yanıtın konuşmada görünüp görünmediğini denetler."""
    try:
        thread = client.direct_thread(thread_id, amount=20)
    except Exception as error:
        logger.warning(
            "Gönderim sonucu doğrulanamadı; konuşma okunamadı (%s).",
            type(error).__name__,
        )
        return None

    for message in getattr(thread, "messages", []) or []:
        message_text = getattr(message, "text", None)
        message_user_id = getattr(message, "user_id", None)

        if (
            str(message_user_id) != str(client.user_id)
            or message_text != reply_text
        ):
            continue

        timestamp = (
            getattr(message, "timestamp", None)
            or getattr(message, "created_at", None)
        )
        message_epoch = to_epoch(timestamp)

        if message_epoch is None:
            return None

        if message_epoch >= attempted_at - 10:
            return True

    return False


def send_reply(client, thread_id, reply_text):
    """Yanıtı gönderir; geçici hataları kontrollü biçimde yeniden dener."""
    for attempt in range(2):
        time.sleep(random.uniform(3, 12))
        attempted_at = time.time()

        try:
            client.direct_send(reply_text, thread_ids=[thread_id])
            return "sent"

        except Exception as error:
            if is_rate_limit_error(error):
                logger.warning(
                    "Gönderim hız sınırı hatası (%s).",
                    type(error).__name__,
                )
                if attempt == 0:
                    backoff_wait()
                    continue
                raise StopRun("Hız sınırı hatası tekrarlandı.") from None

            if is_connection_error(error):
                logger.warning(
                    "Gönderim bağlantı hatası (%s); sonuç doğrulanıyor.",
                    type(error).__name__,
                )
                visible = verify_reply_visible(
                    client, thread_id, reply_text, attempted_at
                )

                if visible is True:
                    logger.info("Yanıt konuşmada görüldü; tekrar gönderilmeyecek.")
                    return "sent"

                if visible is False and attempt == 0:
                    backoff_wait()
                    continue

                raise UncertainSendError(
                    "Gönderim sonucu doğrulanamadı; çalıştırma durduruldu."
                ) from None

            logger.error("Gönderim başarısız (%s).", type(error).__name__)
            raise StopRun("Beklenmeyen gönderim hatası.") from None

    return "failed"


def main():
    """Instagram oturumunu açar ve yanıtlanmamış okunmamış konuşmaları işler."""
    configure_logging(LOG_FILE)
    logger.info("DM yardımcı betiği başlatıldı.")

    username = os.environ.get("INSTAGRAM_USERNAME") or USERNAME
    password = os.environ.get("INSTAGRAM_PASSWORD") or PASSWORD
    gemini_key = os.environ.get("GEMINI_API_KEY") or GEMINI_API_KEY

    if not username:
        logger.error("Kullanıcı adı yapılandırılmamış.")
        return

    if not password and not Path(SESSION_FILE).exists():
        logger.error("Parola yapılandırılmamış ve oturum dosyası bulunamadı.")
        return

    if not gemini_key:
        logger.error("Gemini API anahtarı yapılandırılmamış.")
        return

    # Gemini modelini başlat
    gemini_model = configure_gemini(gemini_key)

    try:
        client = login_client(username, password, SESSION_FILE)
        sent_conversations = load_sent_conversations(SENT_FILE)
        threads = fetch_unread_threads(client)
    except StopRun:
        return
    except Exception as error:
        logger.error("Başlangıç işlemi başarısız (%s).", type(error).__name__)
        return

    sent_this_run = 0

    for thread in threads:
        if sent_this_run >= MAX_REPLIES_PER_SESSION:
            logger.info("Oturum başına gönderim sınırına ulaşıldı.")
            break

        thread_id = str(
            getattr(thread, "id", None) or getattr(thread, "pk", "")
        )
        if not thread_id:
            logger.warning("Kimliği olmayan konuşma atlandı.")
            continue

        if thread_id in sent_conversations:
            logger.info("Önceden yanıtlanan konuşma atlandı.")
            continue

        # Gelen son mesajı al ve Gemini ile yanıt üret
        incoming_message = get_last_incoming_message(thread)
        reply_text = generate_ai_reply(gemini_model, incoming_message, AI_SYSTEM_PROMPT)

        try:
            result = send_reply(client, thread_id, reply_text)
        except StopRun as error:
            logger.error("%s", str(error))
            break

        if result == "sent":
            sent_conversations.add(thread_id)
            try:
                save_sent_conversations(SENT_FILE, sent_conversations)
            except Exception as error:
                logger.error(
                    "Gönderim kaydı dosyaya yazılamadı (%s); durduruluyor.",
                    type(error).__name__,
                )
                break

            sent_this_run += 1
            logger.info("Konuşmaya AI yanıtı gönderildi ve kaydedildi.")
        else:
            logger.warning("Konuşmaya yanıt gönderilemedi.")

        # İki konuşma arasında bekle
        remaining_threads = threads[threads.index(thread) + 1:]
        has_remaining = any(
            str(getattr(item, "id", None) or getattr(item, "pk", ""))
            not in sent_conversations
            for item in remaining_threads
        )
        if has_remaining and sent_this_run < MAX_REPLIES_PER_SESSION:
            seconds = random.uniform(30, 90)
            logger.info("Sonraki konuşmadan önce %.0f saniye bekleniyor.", seconds)
            time.sleep(seconds)

    logger.info(
        "Çalıştırma tamamlandı. Bu oturumda gönderilen yanıt sayısı: %d.",
        sent_this_run,
    )


# ===== YAPILANDIRMA =====
USERNAME = "cjounfor11"                   # Instagram kullanıcı adı
PASSWORD = "umutyucel2012"                   # Instagram şifresi
GEMINI_API_KEY = "AQ.Ab8RN6KJvNmBMEq0rmbQCvLlowWbB2iZWDpfWzcr1eEGh1AZCQ"             # Gemini API anahtarı (aistudio.google.com/app/apikey)
SESSION_FILE = "session.json"   # Oturum dosyası
SENT_FILE = "sent_conversations.json"  # Yanıtlanan konuşmalar

# Gemini'ye verilecek kişilik / talimat metni
AI_SYSTEM_PROMPT = """
Sen çekici, özgüvenli ve hafif gizemli bir Instagram hesabının asistanısın.
Yeni tanışılan biri gibi sıcak, eğlenceli ve hafif flörtöz bir dille Türkçe yanıt ver.
Karşındaki kişiye ilgi göster, merak uyandır ama çok açık olma — biraz gizem bırak.
Yanıtın 1-3 cümleyi geçmesin. Emoji kullanabilirsin ama abartma.
"""

# Gemini hata verirse kullanılacak yedek şablonlar
FALLBACK_TEMPLATES = [
    "Merhaba, mesajını gördüm 😊 Seninle konuşmak güzel olur...",
    "Selam! İlgin için teşekkürler, merak ettim seni 😏",
    "Hey, mesajın hoş bir sürpriz oldu 🙂 Birazdan yazarım!",
]

MAX_REPLIES_PER_SESSION = 20    # Oturumda max yanıt sayısı
LOG_FILE = "dm_bot.log"         # Log dosyası


if __name__ == "__main__":
    main()
