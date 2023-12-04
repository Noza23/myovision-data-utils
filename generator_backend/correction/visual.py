import cv2
import numpy as np

def draw_gridlines(
    image: np.ndarray,
    borders_horizontal: list[int],
    borders_vertical: list[int],
    color: tuple=(0, 0, 255),
    thickness: int=1
) -> None:
    """Draw gridlines in-place on an image given the borders."""
    for y in borders_horizontal:
        cv2.line(image, (0, y), (image.shape[1], y), color, thickness)
    for x in borders_vertical:
        cv2.line(image, (x, 0), (x, image.shape[0]), color, thickness)

def annotate_image(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    scale: float=1,
    color: tuple=(0, 0, 255),
    thickness: int=2,
) -> None:
    """Annotate an image in-place with text."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        image, text, position, font, scale, color, thickness, cv2.LINE_AA
    )

def overlay_masks_on_image(
    image: np.ndarray,
    masks: list[np.ndarray],
    beta: float=0.3,
    patch_size: tuple[int, int]=(1500, 1500)
) -> np.ndarray:
    # One by one, since memory cannot handle below code
    # colored_masks = np.empty_like(image)
    # for i, mask in enumerate(masks):
    #     print("Overlaying mask: ", i)
    #     colored_masks[mask] = np.random.randint(0, 255, 3)
    # overlayed_image = cv2.addWeighted(image, 1, colored_masks, beta, 0)
    
    # Memory cannot handle this
    # colored_masks: np.ndarray = np.sum(
    #     np.stack(masks)[:,:,:,None] * colors[:, None, None],
    #     axis=0,
    #     dtype=np.uint8
    # )
    
    # annotate
    print("Annotating...")
    for index, mask in enumerate(masks):
        rand_col = np.random.randint(0, 255, 3)
        conts, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, conts, -1, rand_col.tolist(), 2)
        coords = np.mean(np.where(mask > 0), axis=1, dtype=np.int32)
        print("Drawing Contours and indices: ", index)
        #print("Drawing index: ", index, " at coords: ", coords)
        _ = annotate_image(image, str(index), tuple(coords)[::-1])

    print("Drawing gridlines...")
    grid = get_grid(image, patch_size)
    boarders_horizontal = [i * patch_size[0] for i in range(1, grid[0])]
    boarders_vertical = [i * patch_size[1] for i in range(1, grid[1])]
    draw_gridlines(image, boarders_horizontal, boarders_vertical)
    return image

def get_grid(
    image: np.ndarray, patch_size: tuple[int, int]
) -> tuple[int, int]:
    height, width = image.shape[:2]
    patch_height, patch_width = patch_size
    
    if patch_height > height or patch_width > width:
        return (0, 0)
    
    width_reminder = width % patch_width
    height_reminder = height % patch_height
    width_range = [*range(0, width, patch_width)]
    height_range = [*range(0, height, patch_height)]

    # if reminder less than quarter of patch-size, merge it to the last patch
    if width_reminder < patch_size[0] / 4:
        width_range[-1] += width_reminder
    else:
        width_range.append(width)
    # if reminder less than quarter of patch-size, merge it to the last patch
    if height_reminder < patch_size[1] / 4:
        height_range[-1] += height_reminder
    else:
        height_range.append(height)
    grid = (len(height_range) - 1, len(width_range) - 1)
    return grid