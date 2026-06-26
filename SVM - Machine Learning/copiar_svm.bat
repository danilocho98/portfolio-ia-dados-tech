@echo off
chcp 65001 >nul
SET OUT=C:\Users\super\Downloads\resultado_portfolio.txt
SET SRC=C:\Users\super\Downloads

FOR /F "tokens=*" %%d IN ('powershell -NoProfile -Command "[Environment]::GetFolderPath(\"Desktop\")"') DO SET DESKTOP=%%d
SET SVM=%DESKTOP%\PROJETO EMPREGO\PORTFOLIO CODIGO\SVM - Machine Learning

echo === Copiando arquivo SVM === > %OUT%
echo SVM destino: %SVM% >> %OUT%
echo. >> %OUT%

echo Arquivos SVM encontrados em Downloads: >> %OUT%
dir "%SRC%\*SVM*" /b >> %OUT% 2>&1
echo. >> %OUT%

echo Copiando... >> %OUT%
for %%f in ("%SRC%\*SVM*") do (
    echo %%~nxf >> %OUT%
    copy /Y "%%f" "%SVM%\" >> %OUT% 2>&1
)

echo. >> %OUT%
echo Conteudo da pasta SVM apos copia: >> %OUT%
dir "%SVM%" /b >> %OUT% 2>&1
echo. >> %OUT%
echo Concluido! >> %OUT%
