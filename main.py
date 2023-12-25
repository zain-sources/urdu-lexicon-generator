from model import G2PModel
model = G2PModel("models")
model.load_decode_model()

print(model.decode(["رکوا", "ایران", "فولاد"]))
