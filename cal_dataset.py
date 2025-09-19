# from datasets import load_dataset

# import os
# # print("========================")
# # print(os.getcwd())


# ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
# # pick first 512 non-empty lines
# calib_lines = [s for s in ds["text"] if s and len(s.strip())>0][:512]
# with open("calib.txt", "w", encoding="utf-8") as f:
#     for s in calib_lines:
#         f.write(s.replace("\n"," ") + "\n")


from datasets import load_dataset
import zstandard as zstd
import json
import os

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

calib_lines = [s for s in ds["text"] if s and len(s.strip()) > 0][:512]
# print(f"len(calib_lines): {len(calib_lines)}") 
# print("==========") 
# print(f"calib_lines[0]: {calib_lines[0]}")
# print("==========")
# print(f"calib_lines[1]: {calib_lines[1]}")
# print("==========")
# print(f"calib_lines[2]: {calib_lines[2]}")
# print("==========")
os.makedirs("dataset", exist_ok=True)

output_path = "dataset/val.jsonl.zst"
cctx = zstd.ZstdCompressor()

with open(output_path, "wb") as f:
    with cctx.stream_writer(f) as writer:
        for line in calib_lines:
            obj = {"text": line.replace("\n", " ")}
            writer.write((json.dumps(obj) + "\n").encode("utf-8"))

print(f"Calibration dataset saved at: {output_path}")
