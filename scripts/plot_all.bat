mkdir out

python .\plotter.py "..\build\CRS_1.csv" "..\build\SOFA_1.csv" "..\build\SoFAM_1.csv" "..\build\MINGO_1.csv" "..\build\DE_1.csv" --title "Bent Cigar func" -o "out\BentCigar_func.jpg"
python .\plotter.py "..\build\CRS_5.csv" "..\build\SOFA_5.csv" "..\build\SoFAM_5.csv" "..\build\MINGO_5.csv" "..\build\DE_5.csv" --title "Rastrigin func" -o "out\Rastrigin_func.jpg"
python .\plotter.py "..\build\CRS_16.csv" "..\build\SOFA_16.csv" "..\build\SoFAM_16.csv" "..\build\MINGO_16.csv" "..\build\DE_16.csv" --title "Hybrid6 func" -o "out\Hybrid6_func.jpg"
python .\plotter.py "..\build\CRS_19.csv" "..\build\SOFA_19.csv" "..\build\SoFAM_19.csv" "..\build\MINGO_19.csv" "..\build\DE_19.csv" --title "Hybrid9 func" -o "out\Hybrid9_func.jpg"
python .\plotter.py "..\build\CRS_28.csv" "..\build\SOFA_28.csv" "..\build\SoFAM_28.csv" "..\build\MINGO_28.csv" "..\build\DE_28.csv" --title "Composition8 func" -o "out\Composit8_func.jpg"
python .\plotter.py "..\build\CRS_30.csv" "..\build\SOFA_30.csv" "..\build\SoFAM_30.csv" "..\build\MINGO_30.csv" "..\build\DE_30.csv" --title "Composition10 func" -o "out\Composit10_func.jpg"

