#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linux計算機サーバーから scp でディレクトリを取得し、
data.txt を整形・グラフ化してメールで送信するスクリプト。

- 1回の実行で「取得 → 集計 → グラフ作成 → メール送信」を行います。
- 定期実行は cron / systemd timer での運用を推奨します
  （本スクリプトにも簡易的な --loop オプションを用意しています）。

必要なパッケージ:
    pip install pandas matplotlib

前提:
    - リモートサーバーへは SSH 鍵認証で scp がパスワード無しで実行できること
      （事前に `ssh-copy-id user@compute-server` 等で設定しておいてください）
    - メール送信は SMTP を利用します（Gmail の場合はアプリパスワードを使用）

設定は環境変数で行います（下記「設定」セクション参照）。
"""

import os
import subprocess
import smtplib
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # GUI無し環境でも動くように
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# 設定（環境変数から読み込み。無い場合はデフォルト値を使用）
# ------------------------------------------------------------------
REMOTE_HOST = os.environ.get("REMOTE_HOST", "takahashi@130.153.73.2")
REMOTE_DIR = os.environ.get("REMOTE_DIR", "~/work/spin1nqs/data/20260724")
LOCAL_WORKDIR = Path(os.environ.get("LOCAL_WORKDIR", "../scp_data"))
DATA_FILENAME = "data.txt"

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
MAIL_FROM = os.environ.get("MAIL_FROM", SMTP_USER)
MAIL_TO = os.environ.get("MAIL_TO", "")

COLUMNS = ["Epoch", "UnixTime", "Re<E>", "Im<E>", "VarE", "Re<S2>", "Im<S2>","<n1>", "<n2>", "<n3>", "n_off"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 1. scp でディレクトリ取得
# ------------------------------------------------------------------
def fetch_remote_dir(remote_host: str, remote_dir: str, local_workdir: Path) -> Path:
    """scp -r でリモートディレクトリをローカルに取得する"""
    local_workdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = local_workdir / timestamp
    dest.mkdir(parents=True, exist_ok=True)

    cmd = ["scp", "-r", f"{remote_host}:{remote_dir}", str(dest)]
    logger.info("実行: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("scp に失敗しました: %s", result.stderr)
        raise RuntimeError(f"scp failed: {result.stderr}")

    # scp -r remote:dir dest の場合、dest/元ディレクトリ名 という構造になる
    subdirs = [p for p in dest.iterdir() if p.is_dir()]
    return subdirs[0] if subdirs else dest


def find_data_file(fetched_dir: Path) -> Path:
    """取得したディレクトリ以下から data.txt を探す"""
    candidates = list(fetched_dir.rglob(DATA_FILENAME))
    if not candidates:
        raise FileNotFoundError(f"{DATA_FILENAME} が見つかりませんでした: {fetched_dir}")
    return candidates[0]


# ------------------------------------------------------------------
# 2. data.txt の読み込み・整形
# ------------------------------------------------------------------
def load_data(data_path: Path) -> pd.DataFrame:
    """
    data.txt を読み込む。
    1行目: ヘッダ (Epoch, Re<E>, Im<E>, VarE, <n1>, <n2>, <n3>, n_off)
    2行目以降: カンマ区切りの数値データ
    """
    df = pd.read_csv(data_path, skiprows=1, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"想定していない列が見つかりません: {missing}. 実際の列: {list(df.columns)}")

    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Epoch"]).sort_values("Epoch").reset_index(drop=True)
    return df


# ------------------------------------------------------------------
# 3. グラフ作成
# ------------------------------------------------------------------
def make_plots(df: pd.DataFrame, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    plot_paths = []

    # (a) epoch - real E
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["Epoch"], df["Re<E>"], color="tab:blue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Energy")
    ax.set_title("Energy (real part) vs Epoch")
    ax.set_ylim(-0.25, -0.2)
    ax.grid(alpha=0.1)
    fig.tight_layout()
    p1 = outdir / "Epoch_vs_energy.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    plot_paths.append(p1)

    # (b) epoch - n1, n2, n3
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["Epoch"], df["<n1>"], label="<n1>")
    ax.plot(df["Epoch"], df["<n2>"], label="<n2>")
    ax.plot(df["Epoch"], df["<n3>"], label="<n3>")
    ax.set_xlabel("epoch")
    ax.set_ylabel("population")
    ax.set_title("<n1>, <n2>, <n3> vs Epoch")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p2 = outdir / "epoch_vs_n1n2n3.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    plot_paths.append(p2)

    # (c) epoch - n off
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["Epoch"], df["n_off"], color="tab:red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("n_off")
    ax.set_title("n_off vs Epoch")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p3 = outdir / "epoch_vs_noff.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    plot_paths.append(p3)

    return plot_paths


# ------------------------------------------------------------------
# 4. メール送信
# ------------------------------------------------------------------
def send_email(subject: str, body: str, image_paths: list[Path], data_path: Path):
    if not (SMTP_USER and SMTP_PASS and MAIL_TO):
        raise RuntimeError(
            "SMTP_USER / SMTP_PASS / MAIL_TO が設定されていません。環境変数を確認してください。"
        )

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = MAIL_FROM
    msg["To"] = MAIL_TO
    msg.attach(MIMEText(body, "plain", "utf-8"))

    for img_path in image_paths:
        with open(img_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-Disposition", "attachment", filename=img_path.name)
            msg.attach(img)

    # 元データも添付しておく
    with open(data_path, "rb") as f:
        att = MIMEApplication(f.read(), Name=data_path.name)
        att["Content-Disposition"] = f'attachment; filename="{data_path.name}"'
        msg.attach(att)

    logger.info("SMTPサーバへ接続: %s:%s", SMTP_HOST, SMTP_PORT)
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
    logger.info("メール送信完了: %s", MAIL_TO)


# ------------------------------------------------------------------
# 5. 一連の処理
# ------------------------------------------------------------------
def run_once():
    logger.info("=== 処理開始 ===")
    fetched_dir = fetch_remote_dir(REMOTE_HOST, REMOTE_DIR, LOCAL_WORKDIR)
    data_path = find_data_file(fetched_dir)
    logger.info("data.txt: %s", data_path)

    df = load_data(data_path)
    logger.info("データ件数: %d 行", len(df))

    plot_dir = fetched_dir / "plots"
    plot_paths = make_plots(df, plot_dir)

    latest_epoch = df["Epoch"].iloc[-1]
    latest_realE = df["Re<E>"].iloc[-1]
    body = (
        f"計算機サーバーから取得したデータの整形結果です。\n\n"
        f"取得元: {REMOTE_HOST}:{REMOTE_DIR}\n"
        f"取得先: {fetched_dir}\n"
        f"データ件数: {len(df)}\n"
        f"最新 epoch: {latest_epoch}\n"
        f"最新 real E: {latest_realE}\n"
    )
    subject = f"[計算結果] epoch={latest_epoch} real E={latest_realE:.6g}"

    send_email(subject, body, plot_paths, data_path)
    logger.info("=== 処理完了 ===")


def main():
    parser = argparse.ArgumentParser(description="scp取得→整形→メール送信スクリプト")
    parser.add_argument(
        "--loop", action="store_true",
        help="指定した間隔で無限ループ実行する（通常運用はcron/systemd timer推奨）"
    )
    parser.add_argument(
        "--interval", type=int, default=3600,
        help="ループ実行時の間隔（秒）。デフォルト3600秒（1時間）"
    )
    args = parser.parse_args()

    if args.loop:
        logger.info("ループモードで起動 (interval=%d秒)", args.interval)
        while True:
            try:
                run_once()
            except Exception:
                logger.exception("処理中にエラーが発生しました")
            time.sleep(args.interval)
    else:
        run_once()


if __name__ == "__main__":
    main()
