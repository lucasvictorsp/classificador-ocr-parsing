"""Constants shared across the document classifier package."""

KNOWN_CLASSES: tuple[str, ...] = (
    "CNH_Frente",
    "CNH_Verso",
    "RG_Frente",
    "RG_Verso",
    "CPF_Frente",
    "CPF_Verso",
)

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
