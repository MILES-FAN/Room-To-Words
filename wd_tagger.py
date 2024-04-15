from tagger.interrogator import Interrogator, WaifuDiffusionInterrogator
from PIL import Image
from pathlib import Path

from tagger.interrogators import interrogators

def image_interrogate(image_path, threshold=0.35, model='wd14-convnextv2.v1'):
    interrogator = interrogators[model]
    im = Image.open(image_path)
    result = interrogator.interrogate(im)
    return Interrogator.postprocess_tags(result[1], threshold)

class wd_tagger:
    def __init__(self, threshold=0.5, model='wd14-convnextv2.v1'):
        self.threshold = threshold
        self.model = model

    def tag_image(self, image_path):
        tags = image_interrogate(Path(image_path), threshold=self.threshold, model=self.model)
        return tags

    def tag_images(self, image_dir, ext='.txt'):
        d = Path(image_dir)
        for f in d.iterdir():
            if not f.is_file() or f.suffix not in ['.png', '.jpg', '.webp']:
                continue
            image_path = Path(f)
            print('processing:', image_path)
            tags = image_interrogate(image_path, threshold=self.threshold, model=self.model)
            tags_str = ", ".join(tags.keys())
            with open(f.parent / f"{f.stem}{ext}", "w") as fp:
                fp.write(tags_str)

    def tag_file(self, image_path):
        tags = image_interrogate(Path(image_path), threshold=self.threshold, model=self.model)
        tags_str = ", ".join(tags.keys())
        print(tags_str)