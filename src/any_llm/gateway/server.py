from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from any_llm.gateway import __version__
from any_llm.gateway.auth.dependencies import set_config
from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.db import get_db, init_db
from any_llm.gateway.pricing_init import initialize_pricing_from_config
from any_llm.gateway.routes import auth, budgets, chat, health, image, keys, pricing, profile, users
from any_llm.gateway.routes.webtoon.panel_dialogue import router as webtoon_panel_dialogue_router
from any_llm.gateway.routes.webtoon.refine_dialogue import router as webtoon_refine_dialogue_router
from any_llm.gateway.routes.webtoon.script import router as webtoon_script_router
from any_llm.gateway.routes.webtoon.topic import router as webtoon_topic_router
from any_llm.gateway.routes.webtoon.topic_from_elements import router as webtoon_topic_from_elements_router
from any_llm.gateway.routes.webtoon.character_sheet import router as webtoon_character_sheet_generation_router
from any_llm.gateway.routes.webtoon.character_sheet_analysis import router as webtoon_character_sheet_analysis_router
from any_llm.gateway.routes.webtoon.caricature_sheet import router as webtoon_caricature_sheet_router
from any_llm.gateway.routes.webtoon.panel_scene import router as webtoon_panel_scene_router
from any_llm.gateway.routes.webtoon.panel_review import router as webtoon_panel_review_router
from any_llm.gateway.routes.webtoon.panel_script import router as webtoon_panel_script_router
from any_llm.gateway.routes.webtoon.review_webtoon import router as webtoon_review_webtoon_router
from any_llm.gateway.routes.webtoon.sns_copy import router as webtoon_sns_copy_router
from any_llm.gateway.routes.webtoon.publish_copy import router as webtoon_publish_copy_router
from any_llm.gateway.routes.webtoon.panel_image import router as webtoon_panel_image_router
from any_llm.gateway.routes.calendar.prompt import router as calendar_prompt_router
from any_llm.gateway.routes.calendar.image import router as calendar_image_router


def _seed_billing_plans(db: "Session") -> None:
    """Ensure billing plans exist (matching careti.ai plans)."""
    from any_llm.gateway.db.caret_models import BillingPlan

    plans = [
        {
            "name": "FREE",
            "monthly_credits": 10.0,
            "monthly_bonus_credits": 20.0,
            "add_amount_usd": 0.0,
            "add_bonus_percent": 0.0,
            "price_usd": 0.0,
            "currency": "USD",
            "credits_per_usd": 10.0,
            "renew_interval_days": 30,
        },
        {
            "name": "BASIC",
            "monthly_credits": 100.0,
            "monthly_bonus_credits": 0.0,
            "add_amount_usd": 10.0,
            "add_bonus_percent": 0.0,
            "price_usd": 10.0,
            "currency": "USD",
            "credits_per_usd": 10.0,
            "renew_interval_days": 30,
        },
    ]

    for plan_data in plans:
        existing = db.query(BillingPlan).filter(
            BillingPlan.name == plan_data["name"],
        ).first()
        if existing:
            for key, value in plan_data.items():
                if key != "name":
                    setattr(existing, key, value)
            existing.active = True
        else:
            db.add(BillingPlan(**plan_data, active=True))
    db.commit()


def create_app(config: GatewayConfig) -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        config: Gateway configuration

    Returns:
        Configured FastAPI application

    """
    init_db(config.database_url, auto_migrate=config.auto_migrate)
    set_config(config)

    db = next(get_db())
    try:
        initialize_pricing_from_config(config, db)
        _seed_billing_plans(db)
    finally:
        db.close()

    app = FastAPI(
        title="any-llm-gateway",
        description="A clean FastAPI gateway for any-llm with API key management",
        version=__version__,
        swagger_ui_parameters={"persistAuthorization": True},
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat.router)
    app.include_router(auth.router)
    app.include_router(keys.router)
    app.include_router(users.router)
    app.include_router(budgets.router)
    app.include_router(pricing.router)
    app.include_router(profile.router)
    app.include_router(health.router)
    app.include_router(image.router)
    app.include_router(webtoon_topic_router)
    app.include_router(webtoon_topic_from_elements_router)
    app.include_router(webtoon_script_router)
    app.include_router(webtoon_panel_dialogue_router)
    app.include_router(webtoon_refine_dialogue_router)
    app.include_router(webtoon_panel_scene_router)
    app.include_router(webtoon_panel_review_router)
    app.include_router(webtoon_panel_script_router)
    app.include_router(webtoon_review_webtoon_router)
    app.include_router(webtoon_sns_copy_router)
    app.include_router(webtoon_publish_copy_router)
    app.include_router(webtoon_panel_image_router)
    app.include_router(webtoon_character_sheet_generation_router)
    app.include_router(webtoon_character_sheet_analysis_router)
    app.include_router(webtoon_caricature_sheet_router)
    app.include_router(calendar_prompt_router)
    app.include_router(calendar_image_router)

    @app.get("/", include_in_schema=False)
    async def redirect_to_docs() -> RedirectResponse:
        """Redirect root requests to interactive API docs."""
        return RedirectResponse(url="/docs")

    return app
