# 股票分析儀表板

這個專案先從第一階段開始：以既有 Excel 範本為核心，每個交易日下載三個來源的每日收盤資料，更新 `1收盤` 工作表的原始資料區，保留原本公式、格式與後續工作表。

## 三階段規劃

1. 第一階段：自動化既有 xlsx
   - 複製範本，不覆蓋原檔。
   - 抓取上市、上櫃、興櫃每日收盤資料。
   - 只更新 `1收盤` 的 raw data 欄位，公式區與篩選表維持原格式。
   - 成交量、成交金額、價格、買賣量等欄位會寫成 Excel 數字型別，方便排序、篩選與公式計算；代號與名稱保留文字。
   - 輸出前會檢查整本活頁簿公式是否與範本完全一致；公式被改動時會停止輸出。
   - 輸出前也會檢查儲存格樣式、合併格、凍結窗格、篩選範圍、欄寬與列高是否與範本一致。
   - 可用 macOS `launchd` 排程週一至週五每日執行。

2. 第二階段：重整資料格式
   - 建立乾淨的長表資料格式，例如 `date, market, symbol, name, open, high, low, close, volume`。
   - 可保留 Excel 輸出，也可改成 SQLite/DuckDB + Streamlit/網頁儀表板。
   - 讓篩選、回測、跨日比較更穩定。

3. 第三階段：統計與機器學習指標
   - 加入移動平均、波動度、成交量異常、趨勢強度、相對強弱等統計指標。
   - 建立特徵工程與模型評估流程。
   - 用可追溯的資料集保存每次指標版本。

## 安裝

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 手動執行

```bash
python src/stage1_close_report.py \
  --template "/Users/allanchang/Downloads/010.2026收盤日報資料整理0713.xlsx"
```

輸出檔會放在 `outputs/YYYYMMDD/`。

## API 設定

主要設定在 `config/sources.json`。等你確認官方 API 後，優先改這個檔案：

- `twse_listed`：上市每日收盤行情
- `tpex_mainboard`：上櫃每日收盤行情
- `tpex_esb`：興櫃當日行情表

目前程式已先支援 JSON 與 CSV 兩種回應格式，且會依欄名對應到範本 `1收盤` 工作表的欄位。

目前 TPEx 免費 OpenAPI 的上櫃與興櫃行情是「最新日」資料；若要回補歷史日期，需改成來源網站實際的歷史 CSV/API。

預設只保留四碼股票代號，避免 ETF、債券、權證等商品讓資料筆數超過原範本公式列。

## 2026-07-13 範本回歸測試

以 `/Users/allanchang/Downloads/010.2026收盤日報資料整理0713.xlsx` 同日測試：

- `上櫃`：使用 TPEx `/www/zh-tw/afterTrading/otc`，`date=115/07/13&type=EW`，`1收盤!A:Q` 逐欄一致。
- `興櫃`：官方當日行情頁目前只找到 `/www/zh-tw/emerging/latest`，不支援指定 115/07/13；歷史日期測試時不覆蓋範本，因此保持一致。
- `上市`：TWSE `MI_INDEX` 可指定 2026-07-13 且欄名一致，但成交股數、成交筆數、成交金額、最後揭示買量/賣量、本益比與範本不同。差異集中在 `C:D:E:M:O:P`，判斷為資料口徑不同。範本看起來較接近每日收盤行情 14:00 產製口徑；TWSE 免費 `STOCK_DAY_ALL`/OpenAPI 較接近此口徑但目前只提供最新日，不能回補 2026-07-13。

下一步需要確認上市可指定日期且符合範本口徑的來源，否則第一階段每日正式跑可以更新，但無法用 7/13 範本做到上市逐欄完全一致。

## 安裝週一至週五排程

先確認 `scripts/install_launchd.sh` 裡的執行時間與範本路徑，再執行：

```bash
bash scripts/install_launchd.sh
```

預設排程為週一至週五 17:30 執行。台股收盤資料通常要等交易所盤後資料發布後才會完整，如果來源延遲，可以把時間改晚。

## GitHub Actions 每日執行

`.github/workflows/daily-stage1.yml` 會在週一至週五台灣時間 16:00 執行，並把產出的 xlsx 上傳為 GitHub Actions artifact。

GitHub runner 看不到本機 `/Users/allanchang/Downloads/...xlsx`，因此不要把範本 xlsx 直接寫死成本機路徑。建議把範本轉成 base64 後放入 repository secret：

```bash
base64 -i "/Users/allanchang/Downloads/010.2026收盤日報資料整理0713.xlsx" | pbcopy
```

接著到 GitHub repository 的 `Settings -> Secrets and variables -> Actions` 新增 secret：

- Name: `STAGE1_TEMPLATE_BASE64`
- Value: 貼上剛剛複製的 base64 內容

手動測試時可到 GitHub Actions 頁面執行 `workflow_dispatch`，可選擇輸入 `trade_date`，例如 `2026-07-13`。正式排程不輸入日期，會使用台灣時間當天。

## 注意

這個階段刻意不改 `日報`、`獸印`、`獨角獸` 等公式工作表，只讓它們延續原本引用邏輯。更新時會依 `1收盤` 既有 raw 代號回填同一列，避免官方資料排序改變造成公式對不齊；來源有但範本沒有的代號會在執行結果中列為提醒。

公式是第一階段的保護重點。程式會在輸出前比對範本與更新後活頁簿的全部公式，包含所有工作表；若公式有任何新增、刪除或變更，會直接報錯，不產出看似成功但公式已偏掉的檔案。

格式同樣會被檢查。程式會比對範本與更新後活頁簿的儲存格樣式，以及工作表層級格式，例如合併儲存格、凍結窗格、篩選範圍、欄寬與列高；若有任何差異，會直接報錯。
