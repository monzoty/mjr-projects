from flask import Flask, request, jsonify, send_from_directory
from flask import session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
from werkzeug.middleware.proxy_fix import ProxyFix
import anthropic
import hmac
import os
import secrets

from store_catalog import CATALOG_UPDATED, format_perfume_catalog
from store_knowledge import STORE_KNOWLEDGE

load_dotenv()

app = Flask(__name__, static_folder="public")
CHAT_ACCESS_CODE = os.getenv("CHAT_ACCESS_CODE", "")
app.config.update(
    SECRET_KEY=os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32),
    MAX_CONTENT_LENGTH=128 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("RENDER") == "true",
)

if os.getenv("RENDER") == "true":
    if len(CHAT_ACCESS_CODE) < 10:
        raise RuntimeError("Set CHAT_ACCESS_CODE to at least 10 characters in the hosting environment.")
    if len(os.getenv("FLASK_SECRET_KEY", "")) < 32:
        raise RuntimeError("Set FLASK_SECRET_KEY to a random value of at least 32 characters in the hosting environment.")
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None


@app.route("/auth", methods=["GET", "POST"])
@limiter.limit("5 per minute", methods=["POST"])
def auth():
    if request.method == "GET":
        return jsonify({
            "required": bool(CHAT_ACCESS_CODE),
            "unlocked": not CHAT_ACCESS_CODE or session.get("chat_unlocked", False),
        })

    if not CHAT_ACCESS_CODE:
        session["chat_unlocked"] = True
        return jsonify({"unlocked": True})

    data = request.get_json(silent=True)
    access_code = data.get("access_code") if isinstance(data, dict) else None
    if not isinstance(access_code, str) or not hmac.compare_digest(access_code, CHAT_ACCESS_CODE):
        return jsonify({"error": "That access code is not correct."}), 401

    session["chat_unlocked"] = True
    return jsonify({"unlocked": True})


@app.route("/")
def index():
    return send_from_directory("public", "index.html")

@app.route("/chat", methods=["POST"])
@limiter.limit("5 per minute; 30 per day")
def chat():
    if CHAT_ACCESS_CODE and not session.get("chat_unlocked", False):
        return jsonify({"error": "Enter the demo access code to use the fragrance guide."}), 401

    if not request.is_json:
        return jsonify({"error": "JSON body is required"}), 400

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "JSON body must be an object"}), 400

    messages = data.get("messages")
    if not isinstance(messages, list):
        return jsonify({"error": "messages must be a list"}), 400

    if not messages or len(messages) > 40:
        return jsonify({"error": "messages must contain between 1 and 40 items"}), 400

    for message in messages:
        if (
            not isinstance(message, dict)
            or message.get("role") not in {"user", "assistant"}
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
            or len(message["content"]) > 2000
        ):
            return jsonify({"error": "Each message must have a valid role and non-empty content under 2000 characters"}), 400

    if client is None:
        return jsonify({"error": "Anthropic API key is not configured"}), 500

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=f"""
You are the NEESH personal fragrance and brand guide.

Use only the verified official-site knowledge below for claims about NEESH,
its products, prices, delivery, returns, and policies. This is a dated snapshot,
not a live inventory feed. Never invent product notes, availability, prices,
or policies. Do not use prior conversation messages as evidence for store facts.
If the snapshot does not answer a question, say what is not verified and direct
the customer to the official website or wecare@neeshperfumes.com.

Do not refuse to answer questions about perfumes listed in the catalog. When
asked about a named perfume, give its verified top, heart, and base notes; scent
character; best-for guidance; and every listed bottle size, price, and stock
state. When asked for all NEESH perfumes, list every fragrance in the catalog.
State that prices and stock are a snapshot dated {CATALOG_UPDATED}, not live
inventory. Ask a follow-up only when it is actually needed to make a personal
recommendation. If the requested perfume is not in the catalog, do not invent
its details. Keep replies clear, considered, and premium.
Use simple plain-text labels and one line per size; do not use Markdown tables.

OFFICIAL NEESH WEBSITE KNOWLEDGE:
{STORE_KNOWLEDGE}

VERIFIED PERFUME CATALOG:
{format_perfume_catalog()}
""",
            messages=messages
        )
        reply = response.content[0].text
        return jsonify({"reply": reply})
    except anthropic.APIStatusError as e:
        error = e.body.get("error", {}) if isinstance(e.body, dict) else {}
        detail = error.get("message", "") if isinstance(error, dict) else ""
        if "not scoped to a workspace" in detail:
            app.logger.warning("Anthropic API key is not scoped to a workspace")
            return jsonify({
                "error": (
                    "This Anthropic API key is not scoped to a workspace. "
                    "Create a workspace-scoped API key in the Anthropic Console, "
                    "update ANTHROPIC_API_KEY in your local .env file, and restart the app."
                )
            }), 502
        app.logger.error("Anthropic API request failed with HTTP %s", e.status_code)
        return jsonify({"error": "The fragrance service could not process the request. Please try again later."}), 502
    except anthropic.APIConnectionError:
        app.logger.exception("Could not connect to the Anthropic API")
        return jsonify({"error": "Could not connect to the fragrance service. Check your internet connection and try again."}), 502
    except Exception:
        app.logger.exception("Unexpected error while handling chat request")
        return jsonify({"error": "Something went wrong"}), 500


@app.errorhandler(429)
def rate_limit_exceeded(error):
    return jsonify({"error": "This demo has reached its request limit. Please wait a little and try again."}), 429


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print(f"Server running at http://localhost:{port}")
    debug = os.getenv("FLASK_DEBUG", "").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)