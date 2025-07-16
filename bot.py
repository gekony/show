import discord
from discord.ext import commands
import cv2
import pytesseract
import re
import os
import csv
from datetime import datetime
import pandas as pd
import numpy as np
from flask import Flask
from threading import Thread

# --- 設定項目 ---
TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
try:
    TARGET_CHANNEL_ID = int(os.environ.get("TARGET_CHANNEL_ID", 0))
except (ValueError, TypeError):
    TARGET_CHANNEL_ID = 0

TEMPLATES_DIR = 'templates'
CSV_FILE = 'drop_data.csv'
STYLE_POINT_TEMPLATE_FILE = 'style_point.png' 
BASE_STYLE_POINT_AMOUNT = 200.0

# 【新機能】アンカーとなるテンプレート画像の名前
ANCHOR_PRIZES_HEADER = 'anchor_prizes_header.png'

# --- Botの初期設定 ---
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

def setup_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['datetime', 'song_name', 'multiplier', 'item_name', 'normalized_amount'])

def extract_normalized_drops(image_path):
    try:
        img_color = cv2.imread(image_path)
        if img_color is None: raise ValueError("画像ファイルが読み込めません")
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(f"画像読み込みエラー: {e}")
        return None

    result = {"song_name": "Unknown", "multiplier": 1.0, "drops": []}

    # --- A. アンカー「獲得プライズ」を探し、各領域を動的に決定 ---
    prizes_area_gray = None
    try:
        anchor_path = os.path.join(TEMPLATES_DIR, ANCHOR_PRIZES_HEADER)
        anchor_template = cv2.imread(anchor_path, 0)
        if anchor_template is None:
            raise FileNotFoundError(f"{ANCHOR_PRIZES_HEADER} が見つかりません。")

        res = cv2.matchTemplate(img_gray, anchor_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val >= 0.8:
            anchor_w, anchor_h = anchor_template.shape[::-1]
            anchor_top_left = max_loc

            # アンカーの位置を基準に、曲名とプライズ領域を決定
            # 1. 曲名領域 (アンカーの上あたり)
            song_y1 = anchor_top_left[1] - 180
            song_y2 = anchor_top_left[1] - 130
            song_x1 = anchor_top_left[0]
            song_x2 = anchor_top_left[0] + 400
            song_roi = img_gray[song_y1:song_y2, song_x1:x2]
            
            # 2. プライズ領域 (アンカーの真下)
            prize_y1 = anchor_top_left[1] + anchor_h
            prize_y2 = prize_y1 + 200 # プライズ領域の高さを200pxと想定
            prize_x1 = anchor_top_left[0] - 100 # 少し左から
            prize_x2 = prize_x1 + 700 # 幅を700pxと想定
            prizes_area_gray = img_gray[prize_y1:prize_y2, prize_x1:prize_x2]

            # 曲名をOCRで読み取る
            song_name_text = pytesseract.image_to_string(song_roi, lang='jpn').strip()
            if song_name_text: result["song_name"] = song_name_text

    except Exception as e:
        print(f"アンカー探索または領域決定でエラー: {e}")
        # アンカーが見つからない場合は、以降の処理をスキップ
        return result

    if prizes_area_gray is None:
        print("プライズ領域を特定できませんでした。")
        return result

    # --- B. プライズ領域内から各アイテムを探す ---
    try:
        # B-1. スタイルポイントを探し、倍率を計算
        sp_template_path = os.path.join(TEMPLATES_DIR, STYLE_POINT_TEMPLATE_FILE)
        sp_template = cv2.imread(sp_template_path, 0)
        if sp_template is not None and not (sp_template.shape[0] > prizes_area_gray.shape[0] or sp_template.shape[1] > prizes_area_gray.shape[1]):
            res = cv2.matchTemplate(prizes_area_gray, sp_template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= 0.8)
            if len(loc[0]) > 0:
                sp_w, sp_h = sp_template.shape[::-1]
                top_left = (loc[1][0], loc[0][0])
                amount_roi = prizes_area_gray[top_left[1] + sp_h - 10 : top_left[1] + sp_h + 40, top_left[0] : top_left[0] + sp_w]
                _, amount_roi_thresh = cv2.threshold(amount_roi, 180, 255, cv2.THRESH_BINARY_INV)
                amount_text = pytesseract.image_to_string(amount_roi_thresh, config="--psm 7 -c tessedit_char_whitelist=x0123456789").strip()
                amount_match = re.search(r'(\d+)', amount_text)
                if amount_match:
                    style_point_amount = int(amount_match.group(1))
                    if style_point_amount > 0: result["multiplier"] = style_point_amount / BASE_STYLE_POINT_AMOUNT

        # B-2. 他の全プライズを探す
        for filename in os.listdir(TEMPLATES_DIR):
            if filename.lower() in [STYLE_POINT_TEMPLATE_FILE.lower(), ANCHOR_PRIZES_HEADER.lower()]: continue
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            
            template_path = os.path.join(TEMPLATES_DIR, filename)
            template = cv2.imread(template_path, 0)
            if template is None or (template.shape[0] > prizes_area_gray.shape[0] or template.shape[1] > prizes_area_gray.shape[1]): continue

            res = cv2.matchTemplate(prizes_area_gray, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= 0.8)
            if len(loc[0]) > 0:
                w, h = template.shape[::-1]
                pt = (loc[1][0], loc[0][0])
                amount_roi = prizes_area_gray[pt[1] + h - 10: pt[1] + h + 40, pt[0]: pt[0] + w]
                _, amount_roi_thresh = cv2.threshold(amount_roi, 180, 255, cv2.THRESH_BINARY_INV)
                amount_text = pytesseract.image_to_string(amount_roi_thresh, config="--psm 7 -c tessedit_char_whitelist=x0123456789").strip()
                amount_match = re.search(r'(\d+)', amount_text)
                if amount_match:
                    found_amount = int(amount_match.group(1))
                    normalized_amount = found_amount / result["multiplier"]
                    item_name = os.path.splitext(filename)[0]
                    result["drops"].append({"item": item_name, "amount": normalized_amount})

    except Exception as e: print(f"プライズ領域の処理でエラー: {e}")
    return result

# --- これ以降の show_stats, on_ready, on_message, Webサーバー機能のコードは変更ありません ---

def show_stats(song_name_filter=None):
    if not os.path.exists(CSV_FILE): return "まだデータがありません。"
    try:
        df = pd.read_csv(CSV_FILE)
        if df.empty: return "データがありません。"
    except pd.errors.EmptyDataError:
        return "データファイルは空です。"
        
    title = "**📊 総合ドロップ率集計**"
    if song_name_filter:
        df = df[df['song_name'].str.contains(song_name_filter, na=False)]
        if df.empty: return f"曲名「{song_name_filter}」を含むデータが見つかりません。"
        title = f"**📊 {song_name_filter} ドロップ率集計**"

    total_runs = df['datetime'].nunique()
    if total_runs == 0: return "集計対象データがありません。"
    
    stats = df.groupby('item_name').agg(drop_runs=('datetime', 'nunique'), total_amount=('normalized_amount', 'sum')).reset_index()
    stats['drop_rate'] = (stats['drop_runs'] / total_runs) * 100
    stats['avg_amount_per_run'] = stats['total_amount'] / total_runs
    
    response = f"{title} (総実行回数: {total_runs}回)\n"
    response += "```\n" + "{:<20} {:<12} {:<15}\n".format("アイテム名", "ドロップ率", "平均ドロップ数/回")
    response += "-"*50 + "\n"
    stats = stats.sort_values(by='drop_rate', ascending=False)
    for _, row in stats.iterrows():
        response += "{:<20} {:>10.2f}% {:>12.2f}\n".format(row['item_name'], row['drop_rate'], row['avg_amount_per_run'])
    response += "```"
    return response

@bot.event
async def on_ready():
    print(f'{bot.user} としてログインしました')
    setup_csv()
    try:
        synced = await bot.tree.sync()
        print(f"{len(synced)}個のコマンドを同期しました")
    except Exception as e: print(e)

@bot.tree.command(name="stats", description="ドロップ率の集計結果を表示します。")
@discord.app_commands.describe(song_name="曲名で結果を絞り込みます（任意）")
async def stats(interaction: discord.Interaction, song_name: str = None):
    stats_text = show_stats(song_name_filter=song_name)
    await interaction.response.send_message(stats_text, ephemeral=True)

@bot.event
async def on_message(message):
    if message.author == bot.user: return
    is_target_channel = (message.channel.id == TARGET_CHANNEL_ID)
    is_dm = isinstance(message.channel, discord.DMChannel)
    if not is_target_channel and not is_dm: return

    if message.attachments:
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith('image/'):
                image_path = f"temp_{attachment.filename}"
                await attachment.save(image_path)
                extracted_data = extract_normalized_drops(image_path)
                os.remove(image_path)
                if extracted_data and extracted_data["drops"]:
                    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        song_name = extracted_data['song_name']
                        for drop in extracted_data["drops"]:
                            writer.writerow([now, song_name, extracted_data['multiplier'], drop['item'], drop['amount']])
                    await message.add_reaction('✅')
                else:
                    await message.add_reaction('❓')

# --- Webサーバー機能で常時起動 ---
app = Flask('')
@app.route('/')
def home():
    return "Bot is alive."
def run():
  app.run(host='0.0.0.0',port=8080)
def keep_alive():
    t = Thread(target=run)
    t.start()

# --- Botの最終実行 ---
if __name__ == "__main__":
    if TOKEN:
        keep_alive()
        bot.run(TOKEN)
    else:
        print("エラー: DISCORD_BOT_TOKENが設定されていません。")
