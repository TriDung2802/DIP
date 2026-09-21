def get_chest_roi(frame,person_box):
    x1,y1,x2,y2=person_box
    person_height = y2 -y1

    chest_y1 = int (y1+person_height*0.25)
    chest_y2 = int (y1+person_height*1.00)

    chest_roi = frame[chest_y1:chest_y2, x1:x2]
    chest_box = (x1,chest_y1,x2,chest_y2)
    return chest_roi,chest_box