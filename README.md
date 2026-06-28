# Papilodem Radiomics Final Projesi

Bu klasor, verilen iki radiomics CSV dosyasi ile papilodem / normal ikili siniflandirma odevi icin hazirlanmis calisilabilir proje iskeletidir.

## 1. Klasor Yapisi

- `data/raw/`: Ham CSV dosyalari.
- `src/radiomics_project/pipeline.py`: Ana makine ogrenmesi pipeline kodu.
- `run_pipeline.py`: Pipeline'i tek komutla calistiran dosya.
- `notebooks/radiomics_final_project.ipynb`: Jupyter Notebook teslim dosyasi icin iskelet.
- `outputs/tables/`: Sonuc tablolari burada olusur.
- `outputs/figures/`: Zorunlu grafikler ve bonus analiz gorselleri burada olusur.
- `report/`: Rapor taslagi ve otomatik PDF raporu burada tutulur.
- `docs/source/`: Odev yonergesi ve workflow gorseli.

## 2. Ortam Kurulumu

Terminalde proje klasorundeyken:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Codex ortamindaki Python 3.12 ile kurmak istersen:

```bash
'/Users/Alp/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3' -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Hızlı Test Calistirmasi

Once kodun calistigini gormek icin kucuk bir deneme yap:

```bash
source .venv/bin/activate
python run_pipeline.py --trials 2 --fast --models LR RF ET GB
```

Bu deneme odev teslimi icin yeterli degildir; sadece kurulum ve kod kontrolu icindir.

## 4. Final Calistirma

Odev yonergesi en az 50 Optuna trial istiyor. Final icin:

```bash
source .venv/bin/activate
python run_pipeline.py --trials 50
```

Bu komut su adimlari uygular:

1. `normal_radiomics.csv` ve `papilodem_radiomics.csv` dosyalarini birlestirir.
2. `Normal = 0`, `Papilledema = 1` etiketlerini olusturur.
3. Hasta seviyesinde train / validation / test ayirimi yapar.
4. Median imputation uygular.
5. Dusuk varyansli ozellikleri siler.
6. Pearson korelasyonu `> 0.95` olan tekrarli ozellikleri eler.
7. `RobustScaler` ile olcekleme yapar.
8. MRMR ozellik secimi uygular.
9. Optuna TPE sampler ile hiperparametre optimizasyonu yapar.
10. LR, SVM, RF, ET, GB, KNN ve MLP modellerini egitir.
11. Sigmoid calibration uygular.
12. RF + ET + GB ile soft voting ensemble olusturur.
13. Weighted ensemble, threshold optimization, nested CV, feature stability, SHAP ve LIME bonus analizlerini uretir.
14. Test metriklerini, grafikleri ve rapor PDF'ini uretir.

## 5. Teslimde Kullanilacak Ciktilar

Final calistirmadan sonra kontrol edecegin dosyalar:

- `outputs/tables/model_performance_test.csv`: Ana test sonuc tablosu.
- `outputs/tables/model_performance_validation.csv`: Validation sonuc tablosu.
- `outputs/tables/best_params.json`: Optuna ile bulunan en iyi parametreler.
- `outputs/tables/top_features.csv`: En onemli radiomics ozellikleri.
- `outputs/tables/statistical_tests.csv`: Friedman, Wilcoxon ve Bonferroni sonuclari.
- `outputs/tables/deep_learning_mlp_results.csv`: MLP bonus model sonucu.
- `outputs/tables/shap_summary.csv`: SHAP aciklanabilirlik sonucu.
- `outputs/tables/lime_explanation.csv`: LIME lokal aciklama sonucu.
- `outputs/tables/nested_cv_results.csv`: Nested cross-validation sonucu.
- `outputs/tables/feature_stability.csv`: Feature stability sonucu.
- `outputs/tables/ensemble_optimization.csv`: Ensemble agirlik optimizasyonu.
- `outputs/tables/threshold_optimization.csv`: Threshold optimization sonucu.
- `outputs/figures/roc_curve.png`
- `outputs/figures/precision_recall_curve.png`
- `outputs/figures/confusion_matrix.png`
- `outputs/figures/feature_importance.png`
- `outputs/figures/calibration_curve.png`
- `outputs/figures/model_comparison.png`
- `outputs/figures/shap_summary.png`
- `outputs/figures/lime_explanation.png`
- `outputs/figures/nested_cv_results.png`
- `outputs/figures/feature_stability.png`
- `outputs/figures/ensemble_optimization.png`
- `outputs/figures/threshold_optimization.png`
- `report/academic_report_with_figures.pdf`

## 6. Raporda Cevaplanacak Sorular

Rapor yazarken su sorulari mutlaka yanitla:

1. Hangi model en iyi performansi verdi?
2. Ensemble model tekil modellerden daha iyi mi?
3. MRMR ozellik secimi performansi artirdi mi?
4. Kalibrasyon model guvenilirligini artirdi mi?
5. ROC-AUC ile PR-AUC arasinda nasil bir iliski gozlemlendi?
6. Veri boyutunun model performansina etkisi nedir?
7. En onemli ilk 10 radiomics ozelligi hangileridir?

## 7. Data Leakage Kontrol Listesi

Teslimden once bunlari kontrol et:

- Test seti sadece en son degerlendirmede kullanildi.
- Ayni hastanin ornekleri farkli splitlere dusmedi.
- Imputation, korelasyon eleme, scaler ve MRMR pipeline icinde fit edildi.
- Optuna skorlamasi train verisindeki inner cross-validation uzerinden yapildi.
- Validation seti sigmoid calibration icin kullanildi.
- Final test metrikleri test setinden bir kez raporlandi.
