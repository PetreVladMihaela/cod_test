import os
import aspose.words as aw


for fileName in os.listdir("Oferte test"):
    doc = aw.Document(os.path.join("Oferte test", fileName))
    doc.save(os.path.join("oferte", fileName[:-5]+".txt"))

for txtFile in os.listdir("oferte"):
    with open(os.path.join("oferte", txtFile), "r+", encoding="utf-8") as f:
        text = f.read()
        request, offer = text.split("I. Scopul documentului:")
        request = request.partition("Solicitarea client:")[2]
        request = request.partition("Oferta pentru firma")[0].strip()
        f.seek(0)
        offer = offer.replace("Evaluation Only. Created with Aspose.Words. Copyright 2003-2024 Aspose Pty Ltd.", "").strip()
        f.write("<s>[INST] "+request+" [/INST] "+offer+" </s>")
        f.truncate()