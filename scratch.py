direction = 0
this_genre = "jazz"
i = 3
sub_id_int = 5

this_label = [direction, this_genre, i, sub_id_int, "dimension 0 is the label, 1 is the genre, 2 is the index into ref_samples of this sample, 3 is the sub ID"]

print(this_label)

this_label.append(3)
print(this_label)
