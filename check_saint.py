import torch

MODEL_DIR = "models"

def inspect(path):
    obj = torch.load(path, map_location="cpu")
    print("\n========================================")
    print("FILE:", path)
    print("TYPE:", type(obj))

    if isinstance(obj, dict):
        print("DICT KEYS:", list(obj.keys())[:30])

        # If it looks like a state_dict
        if all(isinstance(k, str) for k in obj.keys()):
            # show a few tensor shapes
            shown = 0
            for k, v in obj.items():
                if hasattr(v, "shape"):
                    print(f"{k:40s} -> {tuple(v.shape)}")
                    shown += 1
                if shown == 10:
                    break

        # common patterns
        for key in ["state_dict", "model_state_dict", "net", "model"]:
            if key in obj:
                print(f"\nFound '{key}' inside dict. Type:", type(obj[key]))
    else:
        # It might be a full torch model
        try:
            print("It is a torch model. Parameters count:",
                  sum(p.numel() for p in obj.parameters()))
        except Exception as e:
            print("Not able to count parameters:", e)

inspect(f"{MODEL_DIR}/saint_chemical_model.pt")
inspect(f"{MODEL_DIR}/saint_organic_model.pt")