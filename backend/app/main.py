# backend/app/main.py
import os
import datetime as dt
from typing import Optional, List, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response, PlainTextResponse
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel, Field

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Text, and_, text
)
from sqlalchemy.orm import declarative_base, sessionmaker

# =========================
# 데이터베이스 설정 (SQLite 기본)
# =========================
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./trading.db")
engine_kwargs = {}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

# =========================
# 테이블 정의
# =========================
class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)          # buy/sell
    qty = Column(Integer, nullable=False)
    price = Column(Float, nullable=True)               # 지정가(옵션)
    tif = Column(String(10), nullable=True)            # IOC/FOK/GFD 등
    status = Column(String(20), default="pending")     # pending/dispatch/sent/filled/canceled/rejected
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    note = Column(Text, nullable=True)

class Fill(Base):
    __tablename__ = "fills"
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, nullable=False)
    symbol = Column(String(20), nullable=False)
    qty = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    ts = Column(DateTime, default=dt.datetime.utcnow)

class Tick(Base):
    __tablename__ = "ticks"
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    price = Column(Float, nullable=False)
    qty = Column(Integer, nullable=True)
    ts = Column(DateTime, default=dt.datetime.utcnow)

class AgentHealth(Base):
    __tablename__ = "agent_health"
    id = Column(Integer, primary_key=True)
    agent_id = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False)        # online/offline 등
    mode = Column(String(10), nullable=True)           # paper/live
    ts = Column(DateTime, default=dt.datetime.utcnow)

Base.metadata.create_all(engine)

# 인덱스 보강(최초 실행 시 생성, 이후 IF NOT EXISTS로 안전)
def ensure_indexes():
    with engine.begin() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fills_order_id ON fills(order_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fills_ts ON fills(ts)"))

ensure_indexes()

# =========================
# 앱/템플릿
# =========================
app = FastAPI(title="Wins Quant Trading Backend (Lite)")
templates = Jinja2Templates(directory="backend/app/templates")

# =========================
# 유틸
# =========================
def side_sign(side: str) -> int:
    # buy는 현금 유출(음수), sell은 현금 유입(양수)
    return 1 if side.lower() == "sell" else -1

def start_of_day(now: Optional[dt.datetime] = None) -> dt.datetime:
    now = now or dt.datetime.now()
    return dt.datetime(now.year, now.month, now.day)

# =========================
# Pydantic 스키마
# =========================
class HealthResp(BaseModel):
    status: Literal["ok"]

class OrderSubmitReq(BaseModel):
    symbol: str
    side: Literal["buy", "sell"]
    qty: int
    price: Optional[float] = None
    tif: Optional[str] = "IOC"

class OrderSubmitResp(BaseModel):
    order_id: int
    status: str

class NextOrderResp(BaseModel):
    order: Optional[dict] = None

class FillReq(BaseModel):
    order_id: int
    symbol: str
    qty: int
    price: float

class TickReq(BaseModel):
    symbol: str
    price: float
    qty: Optional[int] = None
    ts: Optional[dt.datetime] = None

class AgentHealthReq(BaseModel):
    agent_id: str
    status: str
    mode: Optional[str] = None
    ts: Optional[dt.datetime] = None

class OrderResp(BaseModel):
    id: int
    symbol: str
    side: Literal["buy", "sell"]
    qty: int
    price: Optional[float] = None
    tif: Optional[str] = None
    status: str
    created_at: dt.datetime

class FillItem(BaseModel):
    id: int
    order_id: int
    symbol: str
    qty: int
    price: float
    ts: dt.datetime

class DailyStatsResp(BaseModel):
    date: str = Field(description="YYYY-MM-DD")
    pnl: float = Field(description="당일 실현손익(원)")
    cum_pnl: float = Field(description="누적 실현손익(원)")
    max_drawdown: float = Field(description="당일 최대낙폭(원, 음수 가능)")
    reject_rate: float = Field(ge=0, le=1, description="당일 주문 거부율(0~1)")
    open_orders: int = Field(ge=0, description="현재 미체결/진행 중 주문 수")

class TimelineEvent(BaseModel):
    event_type: Literal["order", "fill"]
    id: int
    ts: dt.datetime
    symbol: str
    side: Optional[str] = None
    qty: Optional[int] = None
    price: Optional[float] = None
    status: Optional[str] = None
    order_id: Optional[int] = None

class AgentHealthLatestResp(BaseModel):
    agent_id: str
    last_ts: dt.datetime
    seconds_since: float
    derived_status: Literal["online", "degraded", "offline"]
    mode: Optional[str] = None

class PageMeta(BaseModel):
    page: int
    size: int
    total: int

class OrdersPage(BaseModel):
    meta: PageMeta
    items: List[OrderResp]

class FillsPage(BaseModel):
    meta: PageMeta
    items: List[FillItem]

# =========================
# 루트/파비콘/헬스
# =========================
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

@app.get("/health", response_model=HealthResp)
def health():
    return {"status": "ok"}

# =========================
# 주문 생성/배포/체결/틱/에이전트 헬스
# =========================
@app.post("/orders/submit", response_model=OrderSubmitResp)
def submit_order(req: OrderSubmitReq):
    with SessionLocal() as db:
        o = Order(
            symbol=req.symbol, side=req.side, qty=int(req.qty),
            price=req.price, tif=req.tif, status="pending",
        )
        db.add(o)
        db.commit()
        db.refresh(o)
        return OrderSubmitResp(order_id=o.id, status=o.status)

@app.get("/orders/next", response_model=NextOrderResp)
def next_order():
    with SessionLocal() as db:
        o = (
            db.query(Order)
            .filter(Order.status == "pending")
            .order_by(Order.created_at.asc(), Order.id.asc())
            .first()
        )
        if not o:
            return NextOrderResp(order=None)
        # 간단 예약(중복 집행 최소 방지)
        o.status = "dispatch"
        db.commit()
        return NextOrderResp(
            order={
                "id": o.id,
                "symbol": o.symbol,
                "side": o.side,
                "qty": o.qty,
                "price": o.price,
                "tif": o.tif,
                "status": o.status,
                "created_at": o.created_at,
            }
        )

@app.post("/fills")
def post_fill(req: FillReq):
    with SessionLocal() as db:
        ord_row = db.query(Order).filter(Order.id == req.order_id).first()
        if not ord_row:
            raise HTTPException(status_code=404, detail="order not found")
        # 체결 기록
        f = Fill(
            order_id=req.order_id,
            symbol=req.symbol,
            qty=int(req.qty),
            price=float(req.price),
            ts=dt.datetime.utcnow(),
        )
        db.add(f)
        # 주문 상태 업데이트(전량 체결 가정)
        ord_row.status = "filled"
        db.commit()
    return {"ok": True}

@app.post("/ticks")
def post_tick(req: TickReq):
    with SessionLocal() as db:
        t = Tick(
            symbol=req.symbol,
            price=float(req.price),
            qty=(int(req.qty) if req.qty is not None else None),
            ts=(req.ts or dt.datetime.utcnow()),
        )
        db.add(t)
        db.commit()
    return {"ok": True}

@app.post("/agent/health")
def post_agent_health(req: AgentHealthReq):
    with SessionLocal() as db:
        h = AgentHealth(
            agent_id=req.agent_id,
            status=req.status,
            mode=req.mode,
            ts=req.ts or dt.datetime.utcnow(),
        )
        db.add(h)
        db.commit()
    return {"ok": True}

@app.get("/agent/health/latest", response_model=AgentHealthLatestResp)
def get_agent_health_latest():
    with SessionLocal() as db:
        rec = db.query(AgentHealth).order_by(AgentHealth.ts.desc()).first()
        if not rec:
            raise HTTPException(status_code=404, detail="no agent health")
        now = dt.datetime.utcnow()
        diff = (now - rec.ts).total_seconds()
        if diff < 10:
            derived = "online"
        elif diff < 30:
            derived = "degraded"
        else:
            derived = "offline"
        return AgentHealthLatestResp(
            agent_id=rec.agent_id,
            last_ts=rec.ts,
            seconds_since=diff,
            derived_status=derived,
            mode=rec.mode
        )

# =========================
# 목록/CSV (경로 충돌 방지: CSV는 정적 경로, 상세는 /id/ 경로)
# =========================
def apply_order_sort(q, sort: str):
    if sort == "created_at_asc":
        return q.order_by(Order.created_at.asc(), Order.id.asc())
    if sort == "id_desc":
        return q.order_by(Order.id.desc())
    if sort == "id_asc":
        return q.order_by(Order.id.asc())
    return q.order_by(Order.created_at.desc(), Order.id.desc())  # 기본: 최신순

@app.get("/orders", response_model=OrdersPage)
def list_orders(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=200),
    symbol: Optional[str] = None,
    status: Optional[str] = None,
    side: Optional[str] = None,
    frm: Optional[str] = None,  # "YYYY-MM-DD" 또는 ISO8601
    to: Optional[str] = None,
    sort: str = Query("created_at_desc")
):
    off = (page - 1) * size
    with SessionLocal() as db:
        conds = []
        if symbol: conds.append(Order.symbol == symbol)
        if status: conds.append(Order.status == status)
        if side:   conds.append(Order.side == side)
        if frm:
            frm_dt = dt.datetime.fromisoformat(frm) if "T" in frm else dt.datetime.fromisoformat(frm + "T00:00:00")
            conds.append(Order.created_at >= frm_dt)
        if to:
            to_dt = dt.datetime.fromisoformat(to) if "T" in to else dt.datetime.fromisoformat(to + "T23:59:59")
            conds.append(Order.created_at <= to_dt)

        q = db.query(Order).filter(and_(*conds)) if conds else db.query(Order)
        total = q.count()
        rows = apply_order_sort(q, sort).offset(off).limit(size).all()
        items = [OrderResp(
            id=o.id, symbol=o.symbol, side=o.side, qty=o.qty,
            price=o.price, tif=o.tif, status=o.status, created_at=o.created_at
        ) for o in rows]
        return OrdersPage(meta=PageMeta(page=page, size=size, total=total), items=items)

@app.get("/orders/export.csv", response_class=PlainTextResponse)
def export_orders_csv(
    symbol: Optional[str] = None,
    status: Optional[str] = None,
    side: Optional[str] = None,
    frm: Optional[str] = None,
    to: Optional[str] = None,
    sort: str = Query("created_at_desc")
):
    with SessionLocal() as db:
        conds = []
        if symbol: conds.append(Order.symbol == symbol)
        if status: conds.append(Order.status == status)
        if side:   conds.append(Order.side == side)
        if frm:
            frm_dt = dt.datetime.fromisoformat(frm) if "T" in frm else dt.datetime.fromisoformat(frm + "T00:00:00")
            conds.append(Order.created_at >= frm_dt)
        if to:
            to_dt = dt.datetime.fromisoformat(to) if "T" in to else dt.datetime.fromisoformat(to + "T23:59:59")
            conds.append(Order.created_at <= to_dt)
        q = db.query(Order).filter(and_(*conds)) if conds else db.query(Order)
        rows = apply_order_sort(q, sort).all()
        lines = ["id,symbol,side,qty,price,tif,status,created_at"]
        for o in rows:
            lines.append(f"{o.id},{o.symbol},{o.side},{o.qty},{o.price or ''},{o.tif or ''},{o.status},{o.created_at.isoformat()}")
        csv = "\n".join(lines)
        return PlainTextResponse(csv, media_type="text/csv")

@app.get("/fills", response_model=FillsPage)
def list_fills(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=200),
    order_id: Optional[int] = None,
    symbol: Optional[str] = None,
    frm: Optional[str] = None,
    to: Optional[str] = None
):
    off = (page - 1) * size
    with SessionLocal() as db:
        conds = []
        if order_id: conds.append(Fill.order_id == order_id)
        if symbol:   conds.append(Fill.symbol == symbol)
        if frm:
            frm_dt = dt.datetime.fromisoformat(frm) if "T" in frm else dt.datetime.fromisoformat(frm + "T00:00:00")
            conds.append(Fill.ts >= frm_dt)
        if to:
            to_dt = dt.datetime.fromisoformat(to) if "T" in to else dt.datetime.fromisoformat(to + "T23:59:59")
            conds.append(Fill.ts <= to_dt)
        q = db.query(Fill).filter(and_(*conds)) if conds else db.query(Fill)
        total = q.count()
        rows = (q.order_by(Fill.ts.desc(), Fill.id.desc())
                  .offset(off).limit(size).all())
        items = [FillItem(
            id=f.id, order_id=f.order_id, symbol=f.symbol, qty=f.qty, price=f.price, ts=f.ts
        ) for f in rows]
        return FillsPage(meta=PageMeta(page=page, size=size, total=total), items=items)

@app.get("/fills/export.csv", response_class=PlainTextResponse)
def export_fills_csv(
    order_id: Optional[int] = None,
    symbol: Optional[str] = None,
    frm: Optional[str] = None,
    to: Optional[str] = None
):
    with SessionLocal() as db:
        conds = []
        if order_id: conds.append(Fill.order_id == order_id)
        if symbol:   conds.append(Fill.symbol == symbol)
        if frm:
            frm_dt = dt.datetime.fromisoformat(frm) if "T" in frm else dt.datetime.fromisoformat(frm + "T00:00:00")
            conds.append(Fill.ts >= frm_dt)
        if to:
            to_dt = dt.datetime.fromisoformat(to) if "T" in to else dt.datetime.fromisoformat(to + "T23:59:59")
            conds.append(Fill.ts <= to_dt)
        q = db.query(Fill).filter(and_(*conds)) if conds else db.query(Fill)
        rows = q.order_by(Fill.ts.desc(), Fill.id.desc()).all()
        lines = ["id,order_id,symbol,qty,price,ts"]
        for f in rows:
            lines.append(f"{f.id},{f.order_id},{f.symbol},{f.qty},{f.price},{f.ts.isoformat()}")
        csv = "\n".join(lines)
        return PlainTextResponse(csv, media_type="text/csv")

# =========================
# 조회(단건) — 경로 조정: /orders/id/{order_id}
# =========================
@app.get("/orders/id/{order_id}", response_model=OrderResp)
def get_order(order_id: int):
    with SessionLocal() as db:
        o = db.query(Order).filter(Order.id == order_id).first()
        if not o:
            raise HTTPException(status_code=404, detail="order not found")
        return OrderResp(
            id=o.id, symbol=o.symbol, side=o.side, qty=o.qty,
            price=o.price, tif=o.tif, status=o.status, created_at=o.created_at
        )

@app.get("/fills/by-order/{order_id}", response_model=List[FillItem])
def get_fills_by_order(order_id: int):
    with SessionLocal() as db:
        rows = (
            db.query(Fill)
            .filter(Fill.order_id == order_id)
            .order_by(Fill.ts.asc())
            .all()
        )
        return [
            FillItem(
                id=f.id, order_id=f.order_id, symbol=f.symbol,
                qty=f.qty, price=f.price, ts=f.ts
            ) for f in rows
        ]

# =========================
# KPI/타임라인
# =========================
@app.get("/stats/daily", response_model=DailyStatsResp)
def get_daily_stats():
    today0 = start_of_day()
    with SessionLocal() as db:
        day_fills = db.query(Fill).filter(Fill.ts >= today0).order_by(Fill.ts.asc()).all()
        all_fills = db.query(Fill).order_by(Fill.ts.asc()).all()

        def side_map(order_ids):
            if not order_ids:
                return {}
            rows = db.query(Order.id, Order.side).filter(Order.id.in_(order_ids)).all()
            return {r[0]: (r[1] or "buy") for r in rows}

        def calc_pnl(fills):
            if not fills:
                return 0.0
            s_map = side_map({f.order_id for f in fills})
            val = 0.0
            for f in fills:
                s = s_map.get(f.order_id, "buy")
                price = float(f.price or 0.0)
                qty = int(f.qty or 0)
                val += side_sign(s) * (qty * price)
            return float(val)

        pnl_today = calc_pnl(day_fills)
        pnl_cum = calc_pnl(all_fills)

        # 당일 Max Drawdown(실현손익 기준 간이)
        eq = peak = max_dd = 0.0
        s_today = side_map({f.order_id for f in day_fills})
        for f in day_fills:
            sign = side_sign(s_today.get(f.order_id, "buy"))
            eq += sign * (int(f.qty or 0) * float(f.price or 0.0))
            if eq > peak:
                peak = eq
            dd = eq - peak
            if dd < max_dd:
                max_dd = dd

        # 당일 거부율
        total_today = db.query(Order).filter(Order.created_at >= today0).count()
        rejected_today = db.query(Order).filter(
            Order.created_at >= today0, Order.status == "rejected"
        ).count()
        reject_rate = (rejected_today / total_today) if total_today > 0 else 0.0

        # 미체결/진행 중
        open_orders = db.query(Order).filter(
            Order.status.in_(["pending", "dispatch", "sent"])
        ).count()

        return DailyStatsResp(
            date=today0.date().isoformat(),
            pnl=round(pnl_today, 2),
            cum_pnl=round(pnl_cum, 2),
            max_drawdown=round(max_dd, 2),
            reject_rate=round(reject_rate, 4),
            open_orders=open_orders,
        )

@app.get("/timeline/recent", response_model=List[TimelineEvent])
def get_timeline_recent(limit: int = Query(50, ge=1, le=200)):
    with SessionLocal() as db:
        recent_orders = db.query(Order).order_by(Order.created_at.desc()).limit(limit).all()
        recent_fills = db.query(Fill).order_by(Fill.ts.desc()).limit(limit).all()

        ord_items = [
            TimelineEvent(
                event_type="order", id=o.id, ts=o.created_at,
                symbol=o.symbol, side=o.side, qty=o.qty, price=o.price, status=o.status
            ) for o in recent_orders
        ]

        ord_map = {}
        if recent_fills:
            order_ids = {f.order_id for f in recent_fills}
            for o in db.query(Order).filter(Order.id.in_(order_ids)).all():
                ord_map[o.id] = o

        fill_items = []
        for f in recent_fills:
            o = ord_map.get(f.order_id)
            fill_items.append(
                TimelineEvent(
                    event_type="fill", id=f.id, ts=f.ts,
                    symbol=f.symbol, side=(o.side if o else None),
                    qty=f.qty, price=f.price, order_id=f.order_id
                )
            )

        merged = ord_items + fill_items
        merged.sort(key=lambda x: x.ts, reverse=True)
        return merged[:limit]

# =========================
# (선택) 간단 블로터 UI
# =========================
@app.get("/blotter", response_class=HTMLResponse)
def blotter(request: Request):
    return templates.TemplateResponse("blotter.html", {"request": request})