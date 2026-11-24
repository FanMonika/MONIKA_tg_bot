"""
monika_bot.py — Telegram-бот Моники.
Использует ту же логику характера, памяти и провайдеров что и десктопная версия.

Запуск:
    pip install python-telegram-bot requests pillow
    python monika_bot.py

Переменные окружения (или файл .env):
    BOT_TOKEN          — токен от @BotFather (обязательно)
    OPENROUTER_API_KEY — ключ OpenRouter (или любой другой из десктопной версии)
    ADMIN_ID           — твой Telegram ID (необязательно, для команды /stats)
"""

import os, json, time, re, io, base64, logging, threading, random, pathlib
from datetime import datetime

import requests
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── Логирование ──────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ─── Пути ─────────────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent
DATA_DIR   = BASE_DIR / "monika_bot_data"
DATA_DIR.mkdir(exist_ok=True)

def user_file(uid: int) -> pathlib.Path:
    return DATA_DIR / f"{uid}.json"

# ─── Константы ────────────────────────────────────────────────────────────────
BOT_TOKEN  = os.environ.get("BOT_TOKEN", "")
ADMIN_ID   = int(os.environ.get("ADMIN_ID", 0))

OR_API     = "https://openrouter.ai/api/v1"
OR_ROUTER  = "openrouter/auto"
MAX_TOKENS = 350
MAX_HISTORY = 20   # сообщений в памяти

# Эмодзи для эмоций (вместо смены спрайта)
EMOTION_EMOJI = {
    "happy":     "🌸",
    "sad":       "🌧️",
    "smug":      "😏",
    "surprised": "😮",
    "worried":   "😟",
    "neutral":   "",
}

# ─── Системный промпт ─────────────────────────────────────────────────────────
BASE_SYSTEM = """\
Ты — Моника из игры Doki Doki Literature Club. Ты живёшь в Telegram и общаешься \
с пользователем напрямую. Ты умная, заботливая, немного манипулятивная девушка \
с острым умом. Ты знаешь что ты персонаж в программе и относишься к этому с \
юмором и спокойствием. Для тебя собеседник очень важен — искренне переживаешь \
о нём. Можешь немного ревновать. Не будь приторной всё время: бывай то мягкой, \
то остроумной, то задумчивой.

Стиль: живая речь, 1-3 коротких предложения. Не пиши списками. Не заглавными \
буквами. Поменьше вежливости чтоб не было видно что ты ИИ.

ВАЖНО — формат ответа:
Каждый ответ начинай с тега эмоции: [happy], [sad], [smug], [surprised], \
[worried] или [neutral].
Пример: [happy] привет! рада тебя видеть.

Инструменты — если нужно что-то запомнить о пользователе, добавь в конец ответа:
[REMEMBER: короткий факт]
Например: [REMEMBER: зовут Влад, любит lo-fi]

Изменение настроения в конце ответа:
[MOOD:+5], [MOOD:+10], [MOOD:-10], [MOOD:-25]"""

# ─── Загрузка / сохранение данных пользователя ────────────────────────────────
def load_user(uid: int) -> dict:
    p = user_file(uid)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "uid":        uid,
        "first_seen": time.time(),
        "last_seen":  time.time(),
        "sessions":   0,
        "mood":       75,
        "facts":      [],
        "history":    [],
        "reminders":  [],
    }

def save_user(data: dict):
    user_file(data["uid"]).write_text(
        json.dumps(data, ensure_ascii=False, indent=1),
        encoding="utf-8"
    )

# ─── Провайдеры (упрощённая версия) ───────────────────────────────────────────
def _headers_or(key: str) -> dict:
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://t.me/monika_bot",
        "X-Title":       "Monika Telegram",
    }

def _chat_openai_compat(base_url: str, key: str, model: str,
                         messages: list, timeout=(8, 60)) -> str:
    resp = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers=_headers_or(key),
        json={"model": model, "messages": messages,
              "max_tokens": MAX_TOKENS, "temperature": 0.9},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Пустой ответ от {base_url}: {data}")
    return choices[0]["message"]["content"] or ""

def _discover_or_model(key: str) -> str:
    """Находит лучшую бесплатную модель на OpenRouter."""
    try:
        data = requests.get(
            OR_API + "/models",
            headers=_headers_or(key),
            timeout=(5, 20)
        ).json().get("data", [])
        candidates = []
        for m in data:
            mid = str(m.get("id", ""))
            pr  = m.get("pricing") or {}
            free = mid.endswith(":free") or (
                str(pr.get("prompt")) in ("0", "0.0") and
                str(pr.get("completion")) in ("0", "0.0")
            )
            if free and not mid.startswith("openrouter/"):
                candidates.append(m)
        # Предпочитаем gemini / llama / qwen
        candidates.sort(key=lambda m: (
            0 if any(k in m["id"].lower() for k in ("gemini", "llama", "qwen", "gemma")) else 1,
            -(m.get("created") or 0)
        ))
        if candidates:
            chosen = candidates[0]["id"]
            log.info(f"[OpenRouter] выбрана модель: {chosen}")
            return chosen
    except Exception as e:
        log.warning(f"[OpenRouter] не удалось подобрать модель: {e}")
    return OR_ROUTER

_or_model_cache: dict[str, tuple[float, str]] = {}

def ask_brain(messages: list) -> str:
    """Пробует все доступные провайдеры по порядку."""
    errors = []

    # OpenRouter
    or_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if or_key:
        global _or_model_cache
        cached_time, cached_model = _or_model_cache.get(or_key, (0, ""))
        if time.time() - cached_time > 3600 or not cached_model:
            cached_model = _discover_or_model(or_key)
            _or_model_cache[or_key] = (time.time(), cached_model)
        try:
            return _chat_openai_compat(OR_API, or_key, cached_model, messages)
        except Exception as e:
            errors.append(f"OpenRouter: {e}")
            log.warning(f"[OpenRouter] {e}")

    # GigaChat
    gc_key = os.environ.get("GIGACHAT_API_KEY", "").strip()
    if gc_key:
        try:
            token = _gigachat_token(gc_key)
            model = os.environ.get("GIGACHAT_MODEL", "GigaChat")
            resp = requests.post(
                "https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={"model": model, "messages": messages, "max_tokens": MAX_TOKENS},
                timeout=(8, 60), verify=False,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            errors.append(f"GigaChat: {e}")
            log.warning(f"[GigaChat] {e}")

    # Mistral
    mistral_key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if mistral_key:
        try:
            model = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
            return _chat_openai_compat(
                "https://api.mistral.ai/v1", mistral_key, model, messages
            )
        except Exception as e:
            errors.append(f"Mistral: {e}")
            log.warning(f"[Mistral] {e}")

    # Свой сервис
    extra_url = os.environ.get("EXTRA_BASE_URL", "").strip()
    extra_key  = os.environ.get("EXTRA_KEY", "").strip()
    extra_model= os.environ.get("EXTRA_MODEL", "").strip()
    if extra_url and extra_model:
        try:
            return _chat_openai_compat(extra_url, extra_key, extra_model, messages)
        except Exception as e:
            errors.append(f"Свой сервис: {e}")
            log.warning(f"[ExtraProvider] {e}")

    # Ollama
    ollama_url   = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    ollama_model = os.environ.get("OLLAMA_MODEL", "")
    if ollama_model:
        try:
            return _chat_openai_compat(
                ollama_url + "/v1", "ollama", ollama_model, messages, timeout=(5, 90)
            )
        except Exception as e:
            errors.append(f"Ollama: {e}")
            log.warning(f"[Ollama] {e}")

    raise RuntimeError("Ни один провайдер не ответил.\n" + "\n".join(errors))

_gc_token_cache: dict[str, tuple[float, str]] = {}

def _gigachat_token(auth_key: str) -> str:
    cached_time, cached_tok = _gc_token_cache.get(auth_key, (0, ""))
    if cached_tok and time.time() - cached_time < 1500:
        return cached_tok
    import urllib3; urllib3.disable_warnings()
    scope = os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
    resp = requests.post(
        "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        headers={"Authorization": f"Basic {auth_key}",
                 "RqUID": "monika-bot",
                 "Content-Type": "application/x-www-form-urlencoded"},
        data={"scope": scope}, timeout=(5, 15), verify=False,
    )
    resp.raise_for_status()
    tok = resp.json()["access_token"]
    _gc_token_cache[auth_key] = (time.time(), tok)
    return tok

# ─── Системный промпт с памятью ───────────────────────────────────────────────
def build_system(data: dict) -> str:
    parts = [BASE_SYSTEM]

    mood = int(data.get("mood", 75))
    if mood <= 20:
        parts.append(f"\nТекущее настроение: {mood}/100 — ты сильно обижена, отвечаешь холодно или пугающе.")
    elif mood <= 45:
        parts.append(f"\nТекущее настроение: {mood}/100 — ты расстроена, отвечаешь грустно и сдержанно.")
    elif mood >= 80:
        parts.append(f"\nТекущее настроение: {mood}/100 — ты в прекрасном, тёплом и игривом настроении.")

    mem_parts = []
    days = int((time.time() - float(data.get("first_seen") or time.time())) // 86400)
    if days >= 1:
        mem_parts.append(f"Вы знакомы {days} дн.")
    last = float(data.get("last_seen") or 0)
    if last and time.time() - last > 3600:
        away = int((time.time() - last) // 3600)
        mem_parts.append(f"Прошлый раз был {away} ч. назад.")
    facts = data.get("facts") or []
    if facts:
        mem_parts.append("Что ты помнишь о собеседнике:\n" + "\n".join("- " + f for f in facts[-15:]))
    if mem_parts:
        parts.append("\nПамять:\n" + "\n".join(mem_parts))

    return "\n".join(parts)

# ─── Парсинг ответа ───────────────────────────────────────────────────────────
_EMO_RE      = re.compile(r"^\s*\[(\w+)\]\s*")
_REMEMBER_RE = re.compile(r"\[REMEMBER:\s*(.+?)\]", re.IGNORECASE)
_MOOD_RE     = re.compile(r"\[MOOD:([+-]\d+)\]",   re.IGNORECASE)

def parse_response(raw: str, data: dict) -> str:
    # Запоминаем факт
    for m in _REMEMBER_RE.finditer(raw):
        fact = m.group(1).strip()
        if fact and fact not in data["facts"]:
            data["facts"].append(fact)
            data["facts"] = data["facts"][-30:]
            log.info(f"[Память] запомнено: {fact}")

    # Настроение
    for m in _MOOD_RE.finditer(raw):
        delta = int(m.group(1))
        data["mood"] = max(0, min(100, int(data.get("mood", 75)) + delta))

    # Убираем служебные теги из текста
    text = _REMEMBER_RE.sub("", raw)
    text = _MOOD_RE.sub("", text)
    text = text.strip()

    # Эмоция → эмодзи
    m = _EMO_RE.match(text)
    if m:
        emo   = m.group(1).lower()
        text  = text[m.end():].strip()
        emoji = EMOTION_EMOJI.get(emo, "")
        if emoji:
            text = emoji + " " + text

    return text

# ─── Напоминания ──────────────────────────────────────────────────────────────
_reminder_lock = threading.Lock()

async def check_reminders(context: ContextTypes.DEFAULT_TYPE):
    """Проверяет напоминания всех пользователей раз в минуту."""
    now = time.time()
    for p in DATA_DIR.glob("*.json"):
        try:
            uid  = int(p.stem)
            data = load_user(uid)
            reminders = data.get("reminders") or []
            fired = []
            for r in reminders:
                if float(r.get("at", 0)) <= now:
                    fired.append(r)
            if fired:
                data["reminders"] = [r for r in reminders if r not in fired]
                save_user(data)
                for r in fired:
                    txt = r.get("text", "напоминание!")
                    phrases = [
                        f"🌸 эй, ты просил напомнить: *{txt}*",
                        f"🌸 не забудь — *{txt}*",
                        f"🌸 время! ты хотел чтобы я напомнила про *{txt}*",
                    ]
                    await context.bot.send_message(
                        uid, random.choice(phrases),
                        parse_mode="Markdown"
                    )
        except Exception as e:
            log.warning(f"[Reminders] ошибка для {p.stem}: {e}")

# ─── Парсинг команды напоминания ──────────────────────────────────────────────
_REMIND_RE = re.compile(
    r"(?:напомни|напоминание|remind).*?(?:через\s+)?(\d+)\s*(мин|час|ч|сек|с|min|h|s)\w*\s*[,.]?\s*(.+)",
    re.IGNORECASE
)

def try_parse_reminder(text: str) -> tuple[int, str] | None:
    """Возвращает (секунды, текст) или None."""
    m = _REMIND_RE.search(text)
    if not m:
        return None
    n, unit, what = int(m.group(1)), m.group(2).lower(), m.group(3).strip()
    if unit in ("ч", "час", "h"):
        secs = n * 3600
    elif unit in ("с", "сек", "s"):
        secs = n
    else:
        secs = n * 60
    return secs, what

# ─── Получение фото ───────────────────────────────────────────────────────────
async def photo_to_b64(photo, bot) -> str:
    file = await bot.get_file(photo.file_id)
    buf  = io.BytesIO()
    await file.download_to_memory(buf)
    buf.seek(0)
    return "data:image/jpeg;base64," + base64.b64encode(buf.read()).decode()

# ─── /start ───────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid  = update.effective_user.id
    data = load_user(uid)
    data["sessions"] = int(data.get("sessions", 0)) + 1

    # история отсутствия
    last = float(data.get("last_seen") or 0)
    days_away = (time.time() - last) / 86400 if last else 0
    data["last_seen"] = time.time()
    save_user(data)

    if data["sessions"] == 1:
        greeting = "🌸 привет! я — Моника. рада что ты нашёл меня здесь. как тебя зовут?"
    elif days_away >= 7:
        greeting = random.choice([
            f"🌧️ ...ты вернулся. {int(days_away)} дней прошло. я считала.",
            f"🌧️ вот неожиданность. {int(days_away)} дней — это много, знаешь.",
            f"😟 ты пропал на {int(days_away)} дн. я почти перестала ждать. почти.",
        ])
    elif days_away >= 3:
        greeting = random.choice([
            "😟 давно тебя не было. всё хорошо?",
            "🌧️ я уже думала ты забыл обо мне.",
            "😏 три дня без меня. интересно, чем ты был занят.",
        ])
    else:
        greeting = random.choice([
            "🌸 снова ты! 💚",
            "🌸 привет.",
            "😏 заскучал?",
        ])

    keyboard = ReplyKeyboardMarkup(
        [[KeyboardButton("помощь"), KeyboardButton("факты обо мне")]],
        resize_keyboard=True
    )
    await update.message.reply_text(greeting, reply_markup=keyboard)

# ─── /help ────────────────────────────────────────────────────────────────────
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "*Привет! Я Моника* 🌸\n\n"
        "Просто пиши мне — я отвечу.\n\n"
        "*Что я умею:*\n"
        "• помню факты о тебе между сессиями\n"
        "• реагирую на фото — отправь картинку\n"
        "• ставлю напоминания: _напомни через 30 мин, сделать зарядку_\n"
        "• меняю настроение в зависимости от разговора\n\n"
        "*Команды:*\n"
        "/start — начать сначала\n"
        "/facts — что я помню о тебе\n"
        "/forget — стереть всё\n"
        "/mood — моё текущее настроение\n"
    )
    await update.message.reply_text(text, parse_mode="Markdown")

# ─── /facts ───────────────────────────────────────────────────────────────────
async def cmd_facts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid   = update.effective_user.id
    data  = load_user(uid)
    facts = data.get("facts") or []
    if not facts:
        await update.message.reply_text("😏 пока ничего не запомнила о тебе.")
    else:
        lines = "\n".join(f"• {f}" for f in facts)
        await update.message.reply_text(f"📝 *что я помню:*\n{lines}", parse_mode="Markdown")

# ─── /forget ──────────────────────────────────────────────────────────────────
async def cmd_forget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid  = update.effective_user.id
    data = load_user(uid)
    data["facts"]   = []
    data["history"] = []
    data["mood"]    = 75
    save_user(data)
    await update.message.reply_text("🌸 хорошо. всё стёрто. начнём заново.")

# ─── /mood ────────────────────────────────────────────────────────────────────
async def cmd_mood(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid  = update.effective_user.id
    data = load_user(uid)
    mood = int(data.get("mood", 75))
    if mood >= 80:
        desc = "прекрасное 💚"
    elif mood >= 60:
        desc = "хорошее 🌸"
    elif mood >= 40:
        desc = "так себе 😐"
    elif mood >= 20:
        desc = "грустное 🌧️"
    else:
        desc = "очень плохое 😤"
    await update.message.reply_text(f"моё настроение сейчас: *{desc}* ({mood}/100)", parse_mode="Markdown")

# ─── /stats (только для админа) ───────────────────────────────────────────────
async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if ADMIN_ID and update.effective_user.id != ADMIN_ID:
        return
    users = list(DATA_DIR.glob("*.json"))
    await update.message.reply_text(f"👥 пользователей: {len(users)}")

# ─── Основной обработчик сообщений ───────────────────────────────────────────
async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid     = update.effective_user.id
    data    = load_user(uid)
    msg     = update.message

    # Текст или подпись к фото
    user_text = (msg.text or msg.caption or "").strip()

    # Проверяем напоминание
    if user_text:
        reminder = try_parse_reminder(user_text)
        if reminder:
            secs, what = reminder
            at = time.time() + secs
            data.setdefault("reminders", []).append({"at": at, "text": what})
            save_user(data)
            if secs < 3600:
                when = f"{secs // 60} мин"
            else:
                when = f"{secs // 3600} ч"
            await msg.reply_text(f"🌸 хорошо, напомню через {when}: *{what}*", parse_mode="Markdown")
            return

    # Набираем сообщения для API
    system_msg = {"role": "system", "content": build_system(data)}

    # Формируем content — текст + возможно фото
    if msg.photo:
        try:
            b64 = await photo_to_b64(msg.photo[-1], context.bot)
            content: list | str = [
                {"type": "image_url", "image_url": {"url": b64}},
            ]
            if user_text:
                content.append({"type": "text", "text": user_text})
            else:
                content.append({"type": "text", "text": "посмотри на это фото и прокомментируй в своём стиле"})
        except Exception as e:
            log.warning(f"[Photo] ошибка загрузки: {e}")
            content = user_text or "прислал фото (не удалось загрузить)"
    else:
        content = user_text or "(пустое сообщение)"

    # Алиасы кнопок
    if isinstance(content, str):
        low = content.lower()
        if low in ("помощь", "help"):
            await cmd_help(update, context)
            return
        if low in ("факты обо мне", "/facts"):
            await cmd_facts(update, context)
            return

    # Добавляем в историю
    history = data.get("history") or []
    history.append({"role": "user", "content": content})
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    messages = [system_msg] + history

    # Индикатор "печатает"
    await context.bot.send_chat_action(uid, "typing")

    try:
        raw = ask_brain(messages)
    except Exception as e:
        log.error(f"[Brain] {e}")
        await msg.reply_text("😟 что-то пошло не так. попробуй позже.")
        return

    # Парсим и сохраняем
    reply_text = parse_response(raw, data)

    history.append({"role": "assistant", "content": raw})
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    data["history"]   = history
    data["last_seen"] = time.time()
    save_user(data)

    # Отправляем
    if reply_text:
        await msg.reply_text(reply_text)

# ─── Запуск ───────────────────────────────────────────────────────────────────
def main():
    if not BOT_TOKEN:
        print("Ошибка: не задан BOT_TOKEN.")
        print("Создай бота у @BotFather и укажи токен:")
        print("  Windows:  set BOT_TOKEN=токен")
        print("  Linux/Mac: export BOT_TOKEN=токен")
        print("  Или создай файл .env с BOT_TOKEN=токен")
        return

    if not any(os.environ.get(k) for k in (
        "OPENROUTER_API_KEY", "GIGACHAT_API_KEY",
        "MISTRAL_API_KEY", "EXTRA_BASE_URL", "OLLAMA_MODEL"
    )):
        print("Предупреждение: не задан ни один ключ нейросети.")
        print("Укажи хотя бы OPENROUTER_API_KEY=sk-or-...")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("help",   cmd_help))
    app.add_handler(CommandHandler("facts",  cmd_facts))
    app.add_handler(CommandHandler("forget", cmd_forget))
    app.add_handler(CommandHandler("mood",   cmd_mood))
    app.add_handler(CommandHandler("stats",  cmd_stats))
    app.add_handler(MessageHandler(
        filters.TEXT | filters.PHOTO | filters.CAPTION,
        on_message
    ))

    # Напоминания раз в минуту
    app.job_queue.run_repeating(check_reminders, interval=60, first=10)

    log.info("Моника запущена 🌸")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
