import nlp as nl
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3, re, io, json
from datetime import datetime, date, timedelta
import plotly.express as px
import plotly.graph_objects as go
from passlib.hash import pbkdf2_sha256
import jwt
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except:
    PROPHET_AVAILABLE = False

DB_FILE = "budgetwise.db"
JWT_SECRET = "secret"
JWT_ALGORITHM = "HS256"
ADMIN_EMAIL = "admin@budgetwise.local"
ADMIN_PASSWORD = "admin123"

DEFAULT_CATS = {
    "Food": ["lunch", "dinner", "restaurant", "cafe", "coffee", "snack", "meal"],
    "Transport": ["bus", "train", "taxi", "uber", "ola", "auto", "cab", "fuel", "petrol"],
    "Shopping": ["amazon", "flipkart", "mall", "clothes", "purchase", "shop", "fashion"],
    "Bills": ["electricity", "wifi", "mobile", "bill", "recharge", "rent", "payment"],
    "Health": ["medicine", "hospital", "doctor", "clinic", "gym", "checkup"],
    "Entertainment": ["movie", "netflix", "spotify", "music", "game", "ticket", "cinema"],
    "Travel": ["flight", "hotel", "trip", "journey", "booking", "tour"],
    "Education": ["course", "book", "school", "college", "exam", "fees"],
    "Income": ["salary", "credited", "bonus", "refund", "cashback"],
    "Others": []
}


def db():
    c = sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    c.row_factory = sqlite3.Row
    return c


def init():
    c = db(); x = c.cursor()
    x.execute(
        "CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT,email TEXT UNIQUE,username TEXT,password_hash TEXT,is_admin INTEGER,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    x.execute(
        "CREATE TABLE IF NOT EXISTS categories(id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT UNIQUE,keywords TEXT)"
    )
    x.execute(
        "CREATE TABLE IF NOT EXISTS transactions(id INTEGER PRIMARY KEY AUTOINCREMENT,user_email TEXT,date DATE,type TEXT,description TEXT,category TEXT,amount REAL,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    x.execute(
        "CREATE TABLE IF NOT EXISTS goals(id INTEGER PRIMARY KEY AUTOINCREMENT,user_email TEXT,goal_amount REAL,target_date DATE,note TEXT,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    c.commit()
    x.execute("SELECT email FROM users WHERE email=?", (ADMIN_EMAIL,))
    if not x.fetchone():
        h = pbkdf2_sha256.hash(ADMIN_PASSWORD)
        x.execute("INSERT INTO users(email,username,password_hash,is_admin) VALUES(?,?,?,1)", (ADMIN_EMAIL, "admin", h))
    for k, v in DEFAULT_CATS.items():
        x.execute("SELECT name FROM categories WHERE name=?", (k,))
        if not x.fetchone():
            x.execute("INSERT INTO categories(name,keywords) VALUES(?,?)", (k, json.dumps(v)))
    c.commit(); c.close()


def jwt_create(e, a=False):
    return jwt.encode({"email": e, "is_admin": a, "iat": int(datetime.utcnow().timestamp())}, JWT_SECRET, algorithm=JWT_ALGORITHM)


def jwt_read(t):
    try:
        return jwt.decode(t, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except:
        return None


def user_add(e, u, p, a=False):
    c = db(); x = c.cursor()
    try:
        h = pbkdf2_sha256.hash(p)
        x.execute("INSERT INTO users(email,username,password_hash,is_admin) VALUES(?,?,?,?)", (e, u, h, int(a)))
        c.commit(); c.close(); return True, None
    except sqlite3.IntegrityError:
        return False, "Email exists"
    except Exception as ex:
        return False, str(ex)


def user_check(e, p):
    c = db(); x = c.cursor()
    x.execute("SELECT * FROM users WHERE email=?", (e,))
    r = x.fetchone(); c.close()
    if not r:
        return False, "No account"
    try:
        if pbkdf2_sha256.verify(p, r["password_hash"]):
            return True, {"email": r["email"], "username": r["username"], "is_admin": bool(r["is_admin"])}
    except:
        return False, "Auth error"
    return False, "Wrong password"


def cats_all():
    c = db(); x = c.cursor()
    x.execute("SELECT name,keywords FROM categories")
    rows = x.fetchall(); c.close()
    out = []
    for r in rows:
        out.append({"name": r["name"], "keywords": json.loads(r["keywords"])})
    return out


def cat_upsert(n, k):
    c = db(); x = c.cursor()
    x.execute("SELECT id FROM categories WHERE name=?", (n,))
    r = x.fetchone()
    if r:
        x.execute("UPDATE categories SET keywords=? WHERE id=?", (json.dumps(k), r["id"]))
    else:
        x.execute("INSERT INTO categories(name,keywords) VALUES(?,?)", (n, json.dumps(k)))
    c.commit(); c.close()


def cat_del(n):
    c = db(); x = c.cursor()
    x.execute("DELETE FROM categories WHERE name=?", (n,))
    c.commit(); c.close()


def auto_cat(d, t):
    d = str(d or "").lower()
    if t == "income":
        return "Income"
    for c in cats_all():
        for k in c["keywords"]:
            if k and re.search(rf"\b{re.escape(k)}\b", d):
                return c["name"]
    return "Others"


def txn_add(u, d, t, des, c, a):
    c1 = db(); x = c1.cursor()
    x.execute("INSERT INTO transactions(user_email,date,type,description,category,amount) VALUES(?,?,?,?,?,?)", (u, d, t, des, c, float(a)))
    c1.commit(); c1.close()


def txn_get(u):
    c = db(); x = c.cursor()
    x.execute("SELECT * FROM transactions WHERE user_email=? ORDER BY date DESC", (u,))
    r = x.fetchall(); c.close()
    if not r:
        return pd.DataFrame(columns=["id", "user_email", "date", "type", "description", "category", "amount", "created_at"])
    df = pd.DataFrame(r, columns=r[0].keys()); df["date"] = pd.to_datetime(df["date"]); return df


def bulk(u, df):
    for _, r in df.iterrows():
        d = r.get("Date") or r.get("date")
        try:
            d2 = pd.to_datetime(d).date()
        except:
            continue

        raw_type = r.get("Type") if r.get("Type") is not None else r.get("type")
        t = str(raw_type).strip().lower() if raw_type is not None else ""
        if t not in ["income", "expense"]:
            amt_val = r.get("Amount") or r.get("amount") or 0
            try:
                amt_val = float(amt_val)
            except:
                amt_val = 0
            t = "income" if amt_val > 0 else "expense"

        ds = r.get("Description") or r.get("description") or ""

        amt_raw = r.get("Amount") if r.get("Amount") is not None else r.get("amount")
        try:
            am = float(amt_raw)
        except:
            am = 0.0

        c = r.get("Category") or r.get("category") or auto_cat(ds, t)

        txn_add(u, d2.isoformat(), t, ds, c, am)


def goal_add(u, a, d, n):
    c = db(); x = c.cursor()
    x.execute("INSERT INTO goals(user_email,goal_amount,target_date,note) VALUES(?,?,?,?)", (u, a, d, n))
    c.commit(); c.close()


def goal_get(u):
    c = db(); x = c.cursor()
    x.execute("SELECT * FROM goals WHERE user_email=? ORDER BY created_at DESC", (u,))
    r = x.fetchall(); c.close()
    if not r:
        return pd.DataFrame(columns=["id", "user_email", "goal_amount", "target_date", "note", "created_at"])
    df = pd.DataFrame(r, columns=r[0].keys()); df["target_date"] = pd.to_datetime(df["target_date"]).dt.date; return df


def stats():
    c = db(); x = c.cursor()
    x.execute("SELECT COUNT(*) c FROM users"); u = x.fetchone()["c"]
    x.execute("SELECT COUNT(*) c FROM transactions"); t = x.fetchone()["c"]
    x.execute("SELECT COUNT(*) c FROM categories"); ca = x.fetchone()["c"]
    c.close(); return {"u": u, "t": t, "c": ca}


# Streamlit session
st.set_page_config(page_title="BudgetWise", layout="wide")
if "auth" not in st.session_state: st.session_state.auth = None
if "email" not in st.session_state: st.session_state.email = None
if "admin" not in st.session_state: st.session_state.admin = False

init()
st.title("💰 BudgetWise")

with st.sidebar:
    if not st.session_state.auth:
        mode = st.selectbox("Account", ["Login", "Signup"])
        if mode == "Signup":
            e = st.text_input("Email"); u = st.text_input("Username"); p = st.text_input("Password", type="password")
            if st.button("Create Account"):
                ok, er = user_add(e, u, p)
                if ok:
                    st.success("Created")
                else:
                    st.error(er)
        else:
            e = st.text_input("Email"); p = st.text_input("Password", type="password")
            if st.button("Login"):
                ok, r = user_check(e, p)
                if ok:
                    st.session_state.auth = jwt_create(r["email"], r["is_admin"])
                    st.session_state.email = r["email"]
                    st.session_state.admin = r["is_admin"]
                    st.rerun()
                else:
                    st.error(r)
    else:
        d = jwt_read(st.session_state.auth)
        st.write(f"👤 {d['email']}")
        if d["is_admin"]: st.write("🛠 Admin")
        if st.button("Logout"):
            st.session_state.auth = None; st.session_state.email = None; st.session_state.admin = False; st.rerun()

if st.session_state.auth:
    menu = ["📊 Dashboard", "📝 Add Transaction", "🔍 Reports", "🔮 Forecast", "🎯 Goals", "👤 Profile"]
    if st.session_state.admin: menu.append("🛠 Admin")
else:
    menu = ["Home"]

m = st.sidebar.radio("Menu", menu)

# ---------------- DASHBOARD ----------------
if m == "Home":
    st.write("Welcome to BudgetWise. Please log in to continue.")

elif m == "📊 Dashboard":
    st.header("📊 Dashboard Overview")
    df = txn_get(st.session_state.email)

    # Custom date filter for dashboard
    st.subheader("Filter by Date Range")
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date", value=df["date"].min() if not df.empty else date.today())
    end_date = col2.date_input("End Date", value=df["date"].max() if not df.empty else date.today())
    if not df.empty:
        df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]

    if df.empty:
        st.info("No transactions yet.")
    else:
        st.subheader("Recent Activity")
        st.dataframe(df[["date", "type", "description", "category", "amount"]].head(10), use_container_width=True)
        inc = df[df.type == "income"].amount.sum()
        exp = df[df.type == "expense"].amount.sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Income", f"₹{inc:,.2f}")
        c2.metric("Total Expense", f"₹{exp:,.2f}")
        c3.metric("Net Balance", f"₹{inc-exp:,.2f}")

        st.subheader("📌 Category Breakdown (Bar Chart)")
        cat = df.groupby("category")["amount"].sum().reset_index()
        if not cat.empty:
            st.plotly_chart(px.bar(cat, x="category", y="amount", title="Spending by Category"), use_container_width=True)

        st.subheader("📈 Daily Trend (Income / Expense / Net)")
        df2 = df.copy()
        df2["date"] = pd.to_datetime(df2["date"])
        d = df2.groupby([pd.Grouper(key="date", freq="D"), "type"])["amount"].sum().reset_index()
        piv = d.pivot(index="date", columns="type", values="amount").fillna(0)
        if "income" not in piv: piv["income"] = 0
        if "expense" not in piv: piv["expense"] = 0
        piv["net"] = piv["income"] - piv["expense"]
        if not piv.empty:
            st.plotly_chart(px.line(piv.reset_index(), x="date", y=["income", "expense", "net"], title="Daily Income/Expense/Net"), use_container_width=True)

# ---------------- ADD TRANSACTION ----------------
elif m == "📝 Add Transaction":
    st.header("📝 Add Transaction")
    e = st.session_state.email
    d = st.date_input("Date")
    ty = st.selectbox("Type", ["expense", "income"])
    ds = st.text_input("Description")
    a = st.number_input("Amount", min_value=0.0)
    c = [x["name"] for x in cats_all()]
    mc = st.selectbox("Category", ["Auto"] + c)
    if st.button("Save"):
        cat = mc if mc != "Auto" else auto_cat(ds, ty)
        txn_add(e, d.isoformat(), ty, ds, cat, a)
        st.success("Transaction Added")
        st.rerun()

    st.subheader("Bulk Upload")
    f = st.file_uploader("Upload CSV")
    if f:
        try:
            dfu = pd.read_csv(f)
            bulk(e, dfu)
            st.success("Uploaded Successfully")
            st.rerun()
        except Exception as ex:
            st.error(f"Bulk upload failed: {ex}")

# ---------------- REPORTS ----------------
elif m == "🔍 Reports":
    st.header("🔍 Spending Reports")
    df = txn_get(st.session_state.email)

    # Custom date filter for reports
    st.subheader("Filter by Date Range")
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date", value=df["date"].min() if not df.empty else date.today(), key="r1")
    end_date = col2.date_input("End Date", value=df["date"].max() if not df.empty else date.today(), key="r2")
    if not df.empty:
        df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]

    if df.empty:
        st.info("No data available.")
    else:
        st.subheader("📌 Expense Distribution (Pie Chart)")
        d = df[df.type == "expense"].groupby("category")["amount"].sum().reset_index()
        st.dataframe(d, use_container_width=True)
        if not d.empty:
            st.plotly_chart(px.pie(d, names="category", values="amount", title="Expense Distribution"), use_container_width=True)

        st.markdown("---")
        st.subheader("📈 Monthly Income vs Expense (Bar Chart)")
        df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
        g = df.groupby(["month", "type"])["amount"].sum().unstack(fill_value=0).reset_index()
        st.dataframe(g, use_container_width=True)
        if not g.empty:
            incs = g["income"] if "income" in g else [0] * len(g)
            exps = g["expense"] if "expense" in g else [0] * len(g)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=g["month"], y=incs, name="Income"))
            fig.add_trace(go.Bar(x=g["month"], y=exps, name="Expense"))
            fig.update_layout(barmode="group", title="Monthly Income vs Expense")
            st.plotly_chart(fig, use_container_width=True)

# ---------------- FORECAST ----------------
elif m == "🔮 Forecast":
    st.header("🔮 Expense Forecasting Engine")
    df = txn_get(st.session_state.email)

    source = st.radio("Select Data Source", ["My Transactions", "Upload CSV"], horizontal=True)
    dff = pd.DataFrame()

    if source == "My Transactions":
        if df.empty:
            st.info("No transactions available.")
        else:
            x = df[df.type == "expense"].copy()
            x["date"] = pd.to_datetime(x["date"])
            dff = x.groupby(pd.Grouper(key="date", freq="D"))["amount"].sum().reset_index()
            dff.rename(columns={"date": "Date", "amount": "Amount"}, inplace=True)
            st.success("Using daily aggregated expense data.")
    else:
        f = st.file_uploader("Upload CSV (Date, Amount)", type=["csv"])
        if f:
            x = pd.read_csv(f)
            x["Date"] = pd.to_datetime(x["Date"], errors="coerce")
            x["Amount"] = pd.to_numeric(x["Amount"], errors="coerce").fillna(0)
            dff = x.groupby(pd.Grouper(key="Date", freq="D"))["Amount"].sum().reset_index()
            st.success("CSV loaded.")

    if not dff.empty:
        st.subheader("Preview (Last 60 Days)")
        st.dataframe(dff.tail(60), use_container_width=True)

    st.subheader("Forecast Duration")
    h = st.radio("Choose Forecast Period", ["Next 1 Month", "Next 6 Months", "Next 12 Months"], horizontal=True)
    days = {"Next 1 Month": 30, "Next 6 Months": 180, "Next 12 Months": 365}[h]

    if st.button("Run Forecast"):
        if dff.empty:
            st.warning("No data available.")
        elif not PROPHET_AVAILABLE:
            st.error("Prophet not installed.")
        else:
            dfp = dff.rename(columns={"Date": "ds", "Amount": "y"})
            dfp["ds"] = pd.to_datetime(dfp["ds"])
            start, end = dfp["ds"].min(), dfp["ds"].max()
            full = pd.DataFrame({"ds": pd.date_range(start, end)})
            dfp = full.merge(dfp, on="ds", how="left").fillna(0)

            valid_rows = dfp[dfp["y"] > 0]
            if len(valid_rows) < 2:
                st.warning("Not enough data to forecast. Add at least 2 days of expense data.")
            else:
                model = Prophet()
                model.fit(dfp)
                fut = model.make_future_dataframe(days)
                fc = model.predict(fut)
                last = dfp["ds"].max()
                fx = fc[fc["ds"] > last][["ds", "yhat"]]
                fx.rename(columns={"ds": "Date", "yhat": "Forecast"}, inplace=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dfp["ds"], y=dfp["y"], name="Actual"))
                fig.add_trace(go.Scatter(x=fx["Date"], y=fx["Forecast"], name="Forecast", line=dict(dash="dash")))
                st.plotly_chart(fig, use_container_width=True)
                st.info(f"Projected Total ({h}): ₹{fx['Forecast'].sum():,.2f}")
                buf = io.StringIO(); fx.to_csv(buf, index=False)
                st.download_button("Download Forecast CSV", buf.getvalue(), "forecast.csv", "text/csv")

# ---------------- GOALS ----------------
elif m == "🎯 Goals":
    st.header("🎯 Financial Goals")
    df = txn_get(st.session_state.email)
    amt = st.number_input("Savings Target (₹)", min_value=0.0)
    td = st.date_input("Target Date")
    note = st.text_input("Note (optional)")
    if st.button("Save Goal"):
        if amt > 0 and td > date.today():
            goal_add(st.session_state.email, amt, td.isoformat(), note)
            st.success("Goal Saved")
            st.rerun()
        else:
            st.warning("Invalid amount or date.")

    g = goal_get(st.session_state.email)
    if not g.empty:
        st.subheader("Your Goals")
        st.dataframe(g, use_container_width=True)

        latest = g.iloc[0]
        ga = float(latest.goal_amount)
        tg = latest.target_date
        dl = (tg - date.today()).days

        if dl > 0:
            if df.empty:
                base = 0
            else:
                dfc = df[df.type == "expense"].copy()
                dfc["date"] = pd.to_datetime(dfc["date"])
                l30 = dfc[dfc["date"] >= pd.Timestamp(date.today()) - pd.Timedelta(days=30)]
                base = l30.amount.sum() / 30 if not l30.empty else dfc.amount.mean()

            need = ga / max(dl, 1)
            st.write(f"Days Remaining: **{dl}**")
            st.write(f"Daily Savings Required: **₹{need:,.2f}**")
            st.write(f"Your Avg Daily Expense: **₹{base:,.2f}**")
            prog = min(100, max(0, (base - need) / base * 100 if base > 0 else 0))
            st.progress(int(prog))

# ---------------- PROFILE ----------------
elif m == "👤 Profile":
    st.header("👤 Profile Settings")
    e = st.session_state.email
    c = db(); x = c.cursor()
    x.execute("SELECT email,username FROM users WHERE email=?", (e,))
    r = x.fetchone(); c.close()
    if r:
        name = st.text_input("Username", r["username"])
        if st.button("Update Profile"):
            c = db(); x = c.cursor()
            x.execute("UPDATE users SET username=? WHERE email=?", (name, e))
            c.commit(); c.close()
            st.success("Profile Updated")
            st.rerun()

# ---------------- ADMIN ----------------
elif m == "🛠 Admin":
    if not st.session_state.admin:
        st.error("Restricted Access")
    else:
        st.header("🛠 Admin Panel")
        s = stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("Users", s["u"])
        c2.metric("Transactions", s["t"])
        c3.metric("Categories", s["c"])

        st.subheader("Manage Categories")
        cs = cats_all()
        nc = st.text_input("Category Name")
        kw = st.text_input("Keywords (comma-separated)")
        if st.button("Save Category"):
            try:
                cat_upsert(nc, [k.strip() for k in kw.split(",") if k.strip()])
                st.success("Saved")
                st.rerun()
            except Exception as ex:
                st.error(f"Save failed: {ex}")

        dc = st.selectbox("Delete Category", [x["name"] for x in cs])
        if st.button("Delete Category"):
            cat_del(dc)
            st.success("Deleted")
            st.rerun()

        # --- Admin analytics and consistency checks ---
        st.markdown("---")
        st.subheader("📈 System Usage Analytics")
        cdb = db()
        daily = pd.read_sql_query("""
            SELECT date, COUNT(*) as count
            FROM transactions
            GROUP BY date
            ORDER BY date
        """, cdb)
        if not daily.empty:
            daily["date"] = pd.to_datetime(daily["date"])
            st.plotly_chart(px.line(daily, x="date", y="count", title="Daily Transaction Volume"), use_container_width=True)

        active = pd.read_sql_query("""
            SELECT user_email, COUNT(*) as txns
            FROM transactions
            GROUP BY user_email
        """, cdb)
        st.write("Active Users:", len(active))

        st.markdown("---")
        st.subheader("🧹 Data Consistency Check")
        checks = {}
        missing_cat = pd.read_sql_query("SELECT * FROM transactions WHERE category IS NULL OR category=''", cdb)
        checks["Missing Category Assignments"] = len(missing_cat)
        negative_amt = pd.read_sql_query("SELECT * FROM transactions WHERE amount < 0", cdb)
        checks["Negative Amounts"] = len(negative_amt)
        dupes = pd.read_sql_query("""
            SELECT user_email, date, description, COUNT(*) as cnt
            FROM transactions
            GROUP BY user_email, date, description
            HAVING cnt > 1
        """, cdb)
        checks["Duplicate Records"] = len(dupes)
        st.write(checks)
        if len(dupes) > 0:
            st.dataframe(dupes, use_container_width=True)

        st.subheader("Recent Transactions")
        df2 = pd.read_sql_query("SELECT * FROM transactions ORDER BY date DESC LIMIT 500", cdb)
        st.dataframe(df2, use_container_width=True)

        st.subheader("All Users")
        df3 = pd.read_sql_query("SELECT email,username,is_admin,created_at FROM users", cdb, parse_dates=["created_at"])
        st.dataframe(df3, use_container_width=True)
        cdb.close()
