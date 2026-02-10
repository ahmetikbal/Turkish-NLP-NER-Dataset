Marmara Üniversitesi NLP (Natural Language Processing) dersi kapsamında geliştirdiğimiz bu proje, 04/2025 tarihi itibariyle bulabildiğimiz tüm Türkçe NER (Named Entity Recognition) datasetlerini birleştirip büyük dil modelleri (LLM) fine-tune etmeyi amaçlamaktadır. Base model olarak Gemma-3-4B-pt tercih edilmiştir. Model eğitimleri Google Colab üzerinden A100 GPU'lar kullanılarak yapılmıştır.

# Dataset: 
https://huggingface.co/datasets/Marmara-NLP/CSE4078S25_Grp1_NER_AlpacaStyle_updated

Huggingface, GitHub, Kaggle gibi kaynaklar taranmış ve aşağıdaki 9 adet Türkçe NER dataseti düzenlemeler yapılarak, standardize edilerek birleştirilmiştir:

| Dataset | Satır sayısı (training dataset) | Satır sayısı (test dataset) | Satır sayısı (validation dataset) | Toplam satır sayısı | URL | Açıklama |
| --- | --- | --- | --- | --- | --- | --- |
| Vitamins and Supplements NER | 2072 | 200 | 200 | 2472 | https://huggingface.co/datasets/turkish-nlp-suite/vitamins-supplements-NER | [Vitaminler.com](http://vitaminler.com/)'dan supplement kullanan müşterilerin yorumları. Yorumlar arasında satın alma nedenleri, etkinlik, dozajlar, yan etkiler, koku, tat vb. yer almaktadır. |
| Turkish Organization NER | 2E+06 | - | - | 1662532 | https://huggingface.co/datasets/STNM-NLPhoenix/turkish-org-ner | Organizasyon varlıklarına odaklanır. 3 etiket vardır: B (Beginning), I (Inside), O (Outside) an organization entity. |
| Turkish Wiki-NER | 18000 | 1000 | 1000 | 20000 | https://github.com/turkish-nlp-suite/Turkish-Wiki-NER-Dataset | Wikipedia cümlelerinden türetilmiş ve Kuzgunlar NER'den yeniden etiketlenmiş bir veri seti. |
| ATISNER (Airline Travel Information System) | 4,978 | 890 | - | 5868 | https://huggingface.co/datasets/ctoraman/atis-ner-turkish | ATISNER, İngilizceden Türkçeye çevrilmiş havayolu  sorguları içerir ve NER için özelleştirilmiştir |
| NER T5 Turkish | - | - | - | 299,800 | https://www.kaggle.com/datasets/binbirmetin/ner-t5-turkish | NER uygulamaları için T5 (a text-to-text transfer transformer) modelini kullanan büyük bir veri kümesi. |
| Turkish NER | - | - | - | 40,000 | https://huggingface.co/datasets/erayyildiz/turkish_ner | Gazeteci kullanılarak otomatik olarak etiketlenmiş Türkçe metin derlemesi. |
| PAN-X.tr | 20000 | 10,000 | 10,000 | 40000 | https://huggingface.co/datasets/xtreme/viewer/PAN-X.tr | MultiNLI metin derlemesi için kitle kaynaklı bir çalışma. |
| NakbaNER | 4032 | - | - | 4032 | https://github.com/sb-b/NakbaTR/tree/main | 1948'de başlayan Filistinlilerin kitlesel göçü olan Nakba'yı konu alan anlatıları yakalamak için geliştirilmiştir. Gerçek tanıklıklar ve haberlerden elde edilmiştir. |
| HisTR | 13100 | 6540 | 5660 | 25306 | https://huggingface.co/datasets/BUCOLIN/HisTR | Servet-i Funun dergisinin sayılarından alınan cümlelerin bir alt kümesini kullanarak elle oluşturulmuş Osmanlı Türkçesi NER veri seti. Edebiyat, bilim, günlük yaşam ve dünya haberleri dahil olmak üzere geniş bir konu yelpazesini kapsamaktadır. |

Final datasetimiz 599.204 adet instancedan oluşuyor. Tamamı full sentence değil ancak büyük çoğunluğu öyle. Örneğin PAN-X.tr 'de cümleler ayrık yapıdaydı ve olduğu gibi tuttuk. turkish-org-ner ise cümle yerine kelime kelime oluşturulmuş bir dataset idi ve bu nedenle 1.6M instancea sahip. Noktalar aracılığıyla parse gerçekleştirip kullandık.
Tüm datasetler aynı labeling taglarını kullanmadığı için standardize ettik. B,I ön eklerini hepsine ekledik. Duplicateleri kaldırdık. Oranı çok az olan classları kaldırdık, onları O'ya dönüştürdük. Son 5 tür classımız ve miktarları bu şekilde:

O: 7879701

B-LOCATION: 288440

I-LOCATION: 54027

B-PERSON: 241500

I-PERSON: 130758

B-ORGANIZATION: 70467

I-ORGANIZATION: 62109

B-DATE: 8673

I-DATE: 7452

B-TITLE: 804165

I-TITLE: 207593


Test setimiz ise 1000 instancetan oluşuyor ve tamamen random şekilde seçildi.


# Sonuçlar ve Proje Raporu

Sonuçlara "Results" klasöründen ve projemizin sonuçları içeren IEEE formatındaki final raporuna aşağıdan erişebilirsiniz:

📄 **[CSE4078S25_Grp1_IEEE_report.pdf](CSE4078S25_Grp1_IEEE_report.pdf)**



# Emeği Geçenler
* Leen I. A. Shaqalaih
* Fatma Melisa Küçük
* Ayşe Sena Aydemir
* Ahmet İkbal Adlığ
* Ahmet Sinan Kalkan


Assoc. Prof. Dr. Murat Can Ganiz'in destekleriyle.
