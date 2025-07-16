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

# --- Botの初期設定 ---
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# --- 既存の関数 (変更なし) ---
def setup_csv():
    # ... (内容は変更なし)
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['datetime', 'song_name', 'multiplier', 'item_name', 'normalized_amount'])

def extract_normalized_drops(image_path):
    print("--- [DEBUG] extract_normalized_drops: 関数を開始 ---")
    try:
        img_color = cv2.imread(image_path)
        if img_color is None: 
            print("[DEBUG] extract_normalized_drops: cv2.imreadで画像が読み込めませんでした。")
            raise ValueError("画像ファイルが読み込めません")
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        print("[DEBUG] extract_normalized_drops: 画像の読み込みとグレースケール化に成功。")
    except Exception as e:
        print(f"[DEBUG] extract_normalized_drops: 画像読み込みエラー: {e}")
        return None

    result = {"song_name": "Unknown", "multiplier": 1.0, "drops": []}

    # A. 曲名を読み取る
    try:
        print("[DEBUG] extract_normalized_drops: 曲名読み取りを開始...")
        song_roi = img_gray[250:300, 300:700]
        song_name_text = pytesseract.image_to_string(song_roi, lang='jpn').strip()
        if song_name_text: 
            result["song_name"] = song_name_text
            print(f"[DEBUG] extract_normalized_drops: 曲名を認識: {song_name_text}")
        else:
            print("[DEBUG] extract_normalized_drops: 曲名は認識されませんでした。")
    except Exception as e: print(f"[DEBUG] extract_normalized_drops: 曲名読み取りでエラー: {e}")

    # B. スタイルポイントを探し、倍率を計算
    try:
        print("[DEBUG] extract_normalized_drops: スタイルポイント検索を開始...")
        sp_template_path = os.path.join(TEMPLATES_DIR, STYLE_POINT_TEMPLATE_FILE)
        sp_template = cv2.imread(sp_template_path, 0)
        sp_w, sp_h = sp_template.shape[::-1]
        res = cv2.matchTemplate(img_gray, sp_template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= 0.8)

        if len(loc[0]) > 0:
            print("[DEBUG] extract_normalized_drops: スタイルポイントのアイコンを発見。")
            top_left = (loc[1][0], loc[0][0])
            amount_roi = img_gray[top_left[1] + sp_h - 10 : top_left[1] + sp_h + 40, top_left[0] : top_left[0] + sp_w]
            _, amount_roi_thresh = cv2.threshold(amount_roi, 180, 255, cv2.THRESH_BINARY_INV)
            amount_text = pytesseract.image_to_string(amount_roi_thresh, config="--psm 7 -c tessedit_char_whitelist=x0123456789").strip()
            amount_match = re.search(r'(\d+)', amount_text)
            if amount_match:
                style_point_amount = int(amount_match.group(1))
                print(f"[DEBUG] extract_normalized_drops: スタイルポイント量を認識: {style_point_amount}")
                if style_point_amount > 0: result["multiplier"] = style_point_amount / BASE_STYLE_POINT_AMOUNT
                print(f"[DEBUG] extract_normalized_drops: 計算後の倍率: {result['multiplier']}")
        else:
            print("[DEBUG] extract_normalized_drops: スタイルポイントのアイコンが見つかりませんでした。")
    except Exception as e: print(f"[DEBUG] extract_normalized_drops: 倍率計算でエラー: {e}")

    # C. 他の全プライズをテンプレートマッチングで探し、正規化して記録
    print("[DEBUG] extract_normalized_drops: 他のプライズの検索を開始...")
    for filename in os.listdir(TEMPLATES_DIR):
        if filename == STYLE_POINT_TEMPLATE_FILE: continue
        try:
            template_path = os.path.join(TEMPLATES_DIR, filename)
            template = cv2.imread(template_path, 0)
            if template is None: continue
            w, h = template.shape[::-1]
            res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= 0.8)
            if len(loc[0]) > 0:
                pt = (loc[1][0], loc[0][0])
                amount_roi = img_gray[pt[1] + h - 10: pt[1] + h + 40, pt[0]: pt[0] + w]
                _, amount_roi_thresh = cv2.threshold(amount_roi, 180, 255, cv2.THRESH_BINARY_INV)
                amount_text = pytesseract.image_to_string(amount_roi_thresh, config="--psm 7 -c tessedit_char_whitelist=x0123456789").strip()
                amount_match = re.search(r'(\d+)', amount_text)
                if amount_match:
                    found_amount = int(amount_match.group(1))
                    normalized_amount = found_amount / result["multiplier"]
                    item_name = os.path.splitext(filename)[0]
                    result["drops"].append({"item": item_name, "amount": normalized_amount})
                    print(f"[DEBUG] extract_normalized_drops: {item_name} を {found_amount}個 発見。 (正規化後: {normalized_amount})")
        except Exception as e: print(f"[DEBUG] extract_normalized_drops: {filename} の処理中にエラー: {e}")
    
    print("--- [DEBUG] extract_normalized_drops: 関数を終了 ---")
    return result

def show_stats(song_name_filter=None):
    # ... (内容は変更なし)
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
    print(f"[DEBUG] 読み込まれたチャンネルID: {TARGET_CHANNEL_ID}") # 起動時にIDを確認
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
    print(f"\n--- [DEBUG] on_message: 新しいメッセージを受信 from {message.author} in channel {message.channel.id} ---")
    if message.author == bot.user: 
        print("[DEBUG] on_message: Bot自身のメッセージのため無視します。")
        return

    is_target_channel = (message.channel.id == TARGET_CHANNEL_ID)
    is_dm = isinstance(message.channel, discord.DMChannel)
    
    print(f"[DEBUG] on_message: is_target_channel={is_target_channel}, is_dm={is_dm}")

    if not is_target_channel and not is_dm: 
        print("[DEBUG] on_message: 対象外のチャンネルのため無視します。")
        return

    if message.attachments:
        print(f"[DEBUG] on_message: {len(message.attachments)}個の添付ファイルを発見。")
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith('image/'):
                print(f"[DEBUG] on_message: 画像ファイル '{attachment.filename}' を処理します。")
                image_path = f"temp_{attachment.filename}"
                
                print(f"[DEBUG] on_message: 画像を '{image_path}' に保存します...")
                await attachment.save(image_path)
                print("[DEBUG] on_message: 保存完了。")

                print("[DEBUG] on_message: extract_normalized_drops関数を呼び出します...")
                extracted_data = extract_normalized_drops(image_path)
                print(f"[DEBUG] on_message: 抽出結果: {extracted_data}")

                print("[DEBUG] on_message: 一時ファイルを削除します...")
                os.remove(image_path)
                print("[DEBUG] on_message: 削除完了。")
                
                if extracted_data and extracted_data["drops"]:
                    print("[DEBUG] on_message: 抽出成功。CSVに書き込みます...")
                    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        song_name = extracted_data['song_name']
                        for drop in extracted_data["drops"]:
                            writer.writerow([now, song_name, extracted_data['multiplier'], drop['item'], drop['amount']])
                    print("[DEBUG] on_message: 書き込み完了。✅リアクションを付けます。")
                    await message.add_reaction('✅')
                else:
                    print("[DEBUG] on_message: 抽出失敗またはドロップなし。❓リアクションを付けます。")
                    await message.add_reaction('❓')
            else:
                print(f"[DEBUG] on_message: 添付ファイル '{attachment.filename}' は画像ではないため無視します。")
    else:
        print("[DEBUG] on_message: 添付ファイルがありません。")

# --- Webサーバー機能 (変更なし) ---
app = Flask('')
@app.route('/')
def home():
    return "Bot is alive."
def run():
  app.run(host='0.0.0.0',port=8080)
def keep_alive():
    t = Thread(target=run)
    t.start()

# --- Botの最終実行 (変更なし) ---
if __name__ == "__main__":
    if TOKEN:
        keep_alive()
        bot.run(TOKEN)
    else:
        print("エラー: DISCORD_BOT_TOKENが設定されていません。")
