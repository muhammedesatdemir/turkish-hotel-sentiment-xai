import pandas as pd

# Dosya yolları
input_path = "./data/processed/final_dataset_merged_3class.csv"
output_path = "./data/processed/final_dataset_no_kaggle_3class.csv"

# CSV'yi oku
df = pd.read_csv(input_path)

# Orijinal satır sayısı
original_count = len(df)

# Kaggle (TripAdvisor) verilerini çıkar
df_filtered = df[df["hotel_name"] != "tripadvisor_hotel"]

# Yeni satır sayısı
filtered_count = len(df_filtered)
removed_count = original_count - filtered_count

# Yeni dosyayı kaydet
df_filtered.to_csv(output_path, index=False, encoding="utf-8")

print("İşlem tamamlandı ✅")
print(f"Orijinal veri sayısı : {original_count}")
print(f"Kalan veri sayısı    : {filtered_count}")
print(f"Çıkarılan satır      : {removed_count}")
print(f"Yeni dosya           : {output_path}")
