from fastapi import FastAPI, Request, File, UploadFile
from fastapi.templating import Jinja2Templates
import io, gc

from generator_backend.utils import (
    encode_image,
    reconstruct_mask_from_patches,
    add_masks,
    save_mask_to_path
)
from generator_backend.model import set_model
from generator_backend.image import set_image
from generator_backend import state

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

print("Loading model...")
_ = set_model("vit_h", "./model/sam_vit_h_4b8939.pth")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/image")
async def upload_image(request: Request, file: UploadFile = File(...)):
    # 1GB limit
    request_data = await request.form()
    patch_size = request_data.get("patch_size")
    width, height = map(int, patch_size.split("x"))
    image_name = file.filename

    if not file.content_type.startswith("image/"):
        return {"error": "Only image files allowed"}
    if file.size > 1e9:
        return {"error": "Image exceed 1GB limit"}
    memory_stream = io.BytesIO()
    for chunk in iter(lambda: file.file.read(65536), b""):
        memory_stream.write(chunk)
    memory_stream.seek(0)
    _ = set_image(memory_stream, image_name, width, height)
    memory_stream.close()
    gc.collect()
    return templates.TemplateResponse(
        "image.html",
        {"request": request, "image_name": image_name}
    )

@app.get("/image/patch/{patch_id}")
def get_patch(request: Request, patch_id: int):
    if not state.IMAGE:
        return {"error": "No image uploaded"}
    n_patches = len(state.IMAGE.patches)
    if patch_id >= n_patches:
        return {
            "error": "Patch ID out of range (0 - {})".format(n_patches - 1)
        }
    patch = state.IMAGE.patches[patch_id]
    instances = [
        add_masks(patch, [inst], make_gray=True)
        for inst in state.IMAGE.masks[patch_id]
    ]
    # save_mask_to_path(patch, path=f"./static/patch.png")
    # instances = state.IMAGE.masks[patch_id]

    patch_url = encode_image(patch)
    instances_urls = [encode_image(inst) for inst in instances]
    if state.IMAGE.valid_instances[patch_id]:
        mask_id = max(state.IMAGE.valid_instances[patch_id])
    else:
        mask_id = 0

    return templates.TemplateResponse(
        "patch.html",
        {
            "request": request,
            "patch": patch_url,
            "masks": instances_urls,
            "patch_id": patch_id,
            "n_patches": n_patches,
            "mask_id": mask_id
        },
    )

@app.post("/add_valid_mask/{patch_id}/{mask_id}")
def add_valid_mask(patch_id: int, mask_id: int):
    if not state.IMAGE:
        return {"error": "No image uploaded"}
    _ = state.IMAGE.add_valid_instance(patch_id, mask_id)
    return {"status": "Button action handled"}


@app.post("/save_mask")
def save_mask():
    # select valid masks
    valid_masks = [
        state.IMAGE.get_valid_mask(patch_id)
        for patch_id in range(len(state.IMAGE.patches))
    ]
    # reconstruct full mask
    mask = reconstruct_mask_from_patches(valid_masks, state.IMAGE.grid)
    # save mask
    fn = state.IMAGE.image_name.split(".")[0] + "_mask.png"
    _ = save_mask_to_path(mask, path=f"./result/{fn}")
    return {"status": f"Mask saved successfully at ./result/{fn}"}


@app.get("/image/state/{patch_id}")
def show_state(request: Request, patch_id: int):
    # Displays All masks on the left and the valid masks on the right
    if state.IMAGE:
        # display all masks
        # masked_image = add_masks(
        #     state.IMAGE.patches[patch_id],
        #     state.IMAGE.masks[patch_id]
        # )
        valids = state.IMAGE.valid_instances[patch_id]
        # display all validated masks
        current_state = add_masks(
            state.IMAGE.patches[patch_id],
            [state.IMAGE.masks[patch_id][mask_id] for mask_id in valids],
        )
        return templates.TemplateResponse(
            "state.html",
            {
                "request": request,
                # "masked_image": encode_image(masked_image),
                "current_state": encode_image(current_state),
            },
        )
    else:
        return {"error": "No image uploaded"}
