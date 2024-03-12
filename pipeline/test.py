import small_fish.pipeline._segmentation as s
import small_fish.gui.prompts as p
import small_fish.pipeline.actions as a

image=[]
label1, label2 = a.launch_segmentation(image)

print(label1)
print(label2)