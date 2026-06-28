# Radyomik Ozellikler Kullanilarak Papilodem Siniflandirmasi

## 1. Giris

Bu calismanin amaci, radyomik ozellikler kullanilarak Normal ve Papilodem siniflarinin makine ogrenmesi yontemleri ile ayrilmasidir. Problem ikili siniflandirma problemidir.

## 2. Veri Seti

Veri seti iki CSV dosyasindan olusmaktadir:

- `normal_radiomics.csv`
- `papilodem_radiomics.csv`

Her ornekte 746 radyomik ozellik bulunmaktadir.
 Hasta bazli veri bolme kullanilmistir; boylece ayni hastaya ait orneklerin farkli train, validation veya test setlerine dusmesi engellenmistir.
Çalışmada toplam 966 örnekten oluşan bir radyomik veri seti kullanılmıştır. Veri setinde 69 hasta bulunmaktadır. Normal sınıfında 672 örnek ve 48 hasta, papilödem sınıfında ise 294 örnek ve 21 hasta yer almaktadır. Her örnek için 746 radyomik özellik bulunmaktadır. Veri setinde eksik değer ve tekrarlı satır bulunmamaktadır.

 Veri seti hasta seviyesinde train, validation ve test alt kümelerine ayrılmıştır. Eğitim setinde 574 örnek ve 41 hasta, validation setinde 196 örnek ve 14 hasta, test setinde ise 196 örnek ve 14 hasta bulunmaktadır. Bu ayrımda aynı hastaya ait örneklerin farklı alt kümelere düşmemesine dikkat edilmiştir.

## 3. Metodoloji

Calismada su pipeline uygulanmistir:

1. Veri setlerinin birlestirilmesi.
2. Sinif etiketlerinin olusturulmasi.
3. Hasta seviyesinde train / validation / test ayrimi.
4. Median imputation.
5. Low-variance filtering.
6. Pearson correlation > 0.95 olan ozelliklerin elenmesi.
7. RobustScaler ile olcekleme.
8. MRMR ozellik secimi.
9. Optuna TPE sampler ile hiperparametre optimizasyonu.
10. Model egitimi.
11. Sigmoid calibration.
12. RF, ET ve GB ile soft voting ensemble.

Bu çalışmada veri sızıntısını önlemek amacıyla tüm ön işleme ve özellik seçimi adımları makine öğrenmesi pipeline'ı içerisinde uygulanmıştır. Median imputation, düşük varyanslı özelliklerin elenmesi, korelasyon temelli özellik azaltma, RobustScaler ile ölçekleme ve MRMR özellik seçimi eğitim verisi üzerinde fit edilmiştir. Test verisi yalnızca final değerlendirme aşamasında kullanılmıştır.

Model seçimi ve hiperparametre optimizasyonu için StratifiedGroupKFold yapısı kullanılmıştır. Böylece sınıf dağılımı korunurken aynı hastaya ait örneklerin farklı foldlara düşmesi engellenmiştir. Optuna TPE sampler ile her model için 50 trial gerçekleştirilmiş ve amaç fonksiyonu Macro-F1 olarak belirlenmiştir. 

Kod açıklamaları ve ilgili dosyalar:

- `load_dataset`: Normal ve papilödem CSV dosyalarını okur, sınıf etiketlerini oluşturur ve iki veri setini birleştirir.
- `make_patient_level_splits`: Aynı hastaya ait örneklerin farklı alt kümelere düşmemesi için hasta seviyesinde train, validation ve test ayrımı yapar.
- `make_preprocessor`: Median imputation, low-variance filtering, Pearson korelasyon eleme ve RobustScaler adımlarını tek bir ön işleme pipeline'ı içinde uygular.
- `MRMRSelector`: Mutual information ve Pearson korelasyon temelli MRMR özellik seçimi yaparak en bilgilendirici radyomik özellikleri seçer.
- `build_pipeline_from_params`: Ön işleme, MRMR özellik seçimi ve makine öğrenmesi modelini tek bir sklearn pipeline yapısında birleştirir.
- `optimize_model`: Optuna TPE sampler ile hiperparametre optimizasyonu yapar ve amaç fonksiyonu olarak Macro-F1 skorunu kullanır.
- `fit_prefit_calibrator`: Validation seti üzerinde sigmoid calibration uygulayarak model olasılık tahminlerini kalibre eder.
- `FittedSoftVotingEnsemble`: RF, ET ve GB modellerinin olasılık çıktılarını ortalayarak soft voting ensemble modeli oluşturur.
- `FittedWeightedSoftVotingEnsemble`: Bonus çalışma kapsamında ensemble üyelerine optimize edilmiş ağırlıklar vererek weighted ensemble modeli oluşturur.
- `metrics_from_proba`: Accuracy, precision, recall, F1, Macro-F1, ROC-AUC, PR-AUC, balanced accuracy ve Brier Score metriklerini hesaplar.
- `run_statistical_tests`: Friedman testi, Wilcoxon signed-rank testi ve Bonferroni düzeltmesi ile modelleri istatistiksel olarak karşılaştırır.
- `run_shap_analysis` ve `run_lime_analysis`: Bonus açıklanabilirlik analizleri ile model kararlarının hangi özelliklerden etkilendiğini gösterir.
- `scripts/build_academic_report_pdf.py`: Oluşturulan tablo ve grafik çıktılarını kullanarak final akademik PDF raporunu üretir.

## 4. Modelleme

Kullanilan modeller:

- Logistic Regression
- RBF SVM
- Random Forest
- Extra Trees
- Gradient Boosting
- K-Nearest Neighbors
- Soft Voting Ensemble

## 5. Hiperparametre Optimizasyonu

Optuna kullanilarak her model icin en az 50 trial denenmistir. Amac fonksiyonu Macro-F1 olarak belirlenmistir. Inner cross-validation yapisinda StratifiedGroupKFold kullanilmistir.

## 6. Sonuclar

Bu bolume `outputs/tables/model_performance_test.csv` tablosundaki sonuclari ekle.

Tablo basliklari:

- Accuracy
- Precision
- Recall
- F1-score
- Macro-F1
- ROC-AUC
- PR-AUC
- Balanced Accuracy
- Brier Score

## 7. Grafikler

Bu bolume asagidaki gorselleri ekle:

- `outputs/figures/roc_curve.png`
- `outputs/figures/precision_recall_curve.png`
- `outputs/figures/confusion_matrix.png`
- `outputs/figures/feature_importance.png`
- `outputs/figures/calibration_curve.png`
- `outputs/figures/model_comparison.png`

## 8. Istatistiksel Analiz

Friedman testi, Wilcoxon signed-rank testi ve Bonferroni duzeltmesi sonuclari `outputs/tables/statistical_tests.csv` dosyasindan rapora eklenmelidir.

## 9. Tartisma

Bu bolumde su sorular cevaplanmalidir:

- En iyi performansi hangi model verdi?
cevap:Final test sonuçlarına göre en yüksek Macro-F1 ve Accuracy değerleri KNN modeli ile elde edilmiştir. Bu nedenle genel sınıflandırma performansı açısından en başarılı model KNN olarak değerlendirilmiştir.
- Ensemble model tekli modellerden daha iyi mi?
cevap:Soft voting ensemble modeli ROC-AUC ve PR-AUC açısından çok güçlü sonuçlar vermiştir. Ancak Macro-F1 açısından KNN modelinin gerisinde kalmıştır. Bu nedenle ensemble modelin olasılık bazlı ayırt edicilikte başarılı olduğu, fakat genel sınıflandırma başarısında en iyi tekil modeli geçemediği söylenebilir.
- MRMR ozellik secimi performansa nasil etki etti?
cevap:MRMR özellik seçimi, yüksek boyutlu 746 radyomik özellik arasından daha bilgilendirici ve daha az tekrar eden özelliklerin seçilmesini sağlamıştır. Bu yaklaşım modelin gereksiz ve yüksek korelasyonlu özelliklerden etkilenmesini azaltarak daha kararlı bir öğrenme süreci sağlamıştır.
- Kalibrasyon model guvenilirligini artirdi mi?
cevap:Sigmoid kalibrasyon, modellerin olasılık çıktılarının daha güvenilir hale getirilmesi amacıyla uygulanmıştır. Brier Score değerlerinin genel olarak düşük seviyede olması, kalibrasyon sonrasında olasılık tahminlerinin makul düzeyde güvenilir olduğunu göstermektedir.
- ROC-AUC ve PR-AUC sonuclari arasinda nasil bir fark goruldu?
cevap:ROC-AUC değerleri modellerin sınıfları genel olarak iyi ayırabildiğini göstermiştir. PR-AUC değerleri özellikle papilödem sınıfının daha az sayıda örnek içermesi nedeniyle önemlidir. Random Forest ve ensemble modellerinin hem ROC-AUC hem de PR-AUC değerlerinin yüksek olması, bu modellerin sınıf ayrımında ve pozitif sınıfı yakalamada güçlü olduğunu göstermektedir.

- Yuksek boyutlu veri yapisi model performansini nasil etkiledi?
cevap:Veri setinde 746 radyomik özellik bulunması, örnek sayısına göre yüksek boyutlu bir yapı oluşturmaktadır. Bu durum aşırı öğrenme riskini artırabilir. Bu nedenle düşük varyanslı özelliklerin silinmesi, korelasyon temelli eleme ve MRMR özellik seçimi gibi boyut azaltıcı adımlar model performansı ve genellenebilirlik açısından kritik öneme sahiptir.

- En onemli 10 radiomics ozelligi hangileridir?
cevap:Feature importance analizine göre en önemli özellikler Feature_0005, Feature_0040, Feature_0436, Feature_0224, Feature_0570, Feature_0670, Feature_0592, Feature_0164, Feature_0022 ve Feature_0478 olarak belirlenmiştir.

Final feature importance analizinde en yüksek katkıya sahip özellik Feature_0005 olmuştur (importance = 0.658023). Bu özelliği Feature_0040 (0.049211), Feature_0436 (0.034976), Feature_0224 (0.031259) ve Feature_0570 (0.030640) izlemiştir. Feature_0005'in diğer özelliklere göre belirgin şekilde yüksek önem değerine sahip olması, model kararlarında bu radyomik özelliğin baskın bir rol oynadığını göstermektedir.

Özellik         Importance
Feature_0005    0.658023
Feature_0040    0.049211
Feature_0436    0.034976
Feature_0224    0.031259
Feature_0570    0.030640
Feature_0670    0.026141
Feature_0592    0.025970
Feature_0164    0.023599
Feature_0022    0.022127
Feature_0478    0.014496


## 10. Sonuc

Bu calismada veri sizintisini engelleyen hasta seviyesinde bir makine ogrenmesi pipeline'i kurulmustur. Elde edilen sonuclar, radyomik ozelliklerin papilodem siniflandirmasinda kullanilabilirligini degerlendirmek icin raporlanmistir.

Final test seti sonuçlarına göre en yüksek Macro-F1 skoru KNN modeli ile elde edilmiştir (Macro-F1 = 0.9133). KNN modeli ayrıca en yüksek doğruluk değerini vermiştir (Accuracy = 0.9337). Bu sonuç, KNN modelinin test setindeki genel sınıflandırma performansının diğer modellere göre daha güçlü olduğunu göstermektedir.

ROC-AUC açısından en başarılı model Random Forest olmuştur (ROC-AUC = 0.9809). PR-AUC metriğinde de en yüksek değer Random Forest modeli ile elde edilmiştir (PR-AUC = 0.9583). Bu durum Random Forest modelinin olasılık skorlarını sınıfları ayıracak şekilde güçlü biçimde sıralayabildiğini göstermektedir.

RF, ET ve GB modellerinden oluşturulan soft voting ensemble modeli de güçlü bir performans göstermiştir (Macro-F1 = 0.8896, ROC-AUC = 0.9805, PR-AUC = 0.9569). Ancak ensemble model, Macro-F1 açısından en iyi tekil model olan KNN'yi geçememiştir. Buna karşın ROC-AUC ve PR-AUC değerleri Random Forest sonucuna oldukça yakın olduğu için ensemble modelin olasılık bazlı ayırt edicilik açısından başarılı olduğu söylenebilir.

Bu çalışmada radyomik özellikler kullanılarak normal ve papilödem sınıflarını ayırmaya yönelik hasta seviyesinde veri sızıntısını önleyen bir makine öğrenmesi pipeline'ı geliştirilmiştir. Veri ön işleme, düşük varyans ve korelasyon temelli özellik eleme, MRMR özellik seçimi, Optuna tabanlı hiperparametre optimizasyonu, sigmoid kalibrasyon ve soft voting ensemble adımları uygulanmıştır.

Final test sonuçlarına göre genel sınıflandırma performansı açısından en başarılı model KNN olmuştur. Random Forest modeli ise ROC-AUC ve PR-AUC metriklerinde en yüksek sonuçları vermiştir. Ensemble model güçlü olasılık bazlı ayırt edicilik göstermesine rağmen Macro-F1 açısından en iyi tekil model olan KNN'yi geçememiştir.

Elde edilen bulgular, radyomik özelliklerin papilödem sınıflandırmasında anlamlı ayırt edici bilgi taşıdığını göstermektedir. Bununla birlikte veri setinin yüksek boyutlu olması nedeniyle özellik seçimi ve hasta seviyesinde doğrulama adımları modelin güvenilirliği açısından kritik öneme sahiptir.

Model      Accuracy   Macro-F1   ROC-AUC   PR-AUC   Brier Score
LR         0.9235     0.8973     0.9459    0.9167   0.0635
SVM        0.9286     0.9048     0.9328    0.9067   0.0643
RF         0.8980     0.8580     0.9809    0.9583   0.0737
ET         0.9286     0.9048     0.9700    0.9464   0.0578
GB         0.9133     0.8866     0.9573    0.9272   0.0662
KNN        0.9337     0.9133     0.9296    0.9050   0.0624
Ensemble   0.9184     0.8896     0.9805    0.9569   0.0617


## 11. Kaynakca


- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
- Akiba, T. et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. Proceedings of the 25th ACM SIGKDD International Conference.
- van Griethuysen, J. J. M. et al. (2017). Computational Radiomics System to Decode the Radiographic Phenotype. Cancer Research, 77(21), e104-e107.
- Gillies, R. J., Kinahan, P. E., & Hricak, H. (2016). Radiomics: Images Are More than Pictures, They Are Data. Radiology, 278(2), 563-577.
- scikit-learn documentation: https://scikit-learn.org/
- Optuna documentation: https://optuna.org/ 
