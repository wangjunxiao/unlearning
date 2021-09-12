list_classes = [6, 9, 9, 4, 1, 1, 2, 7, 8, 3]
list_features = [10, 20, 30, 40, 50, 60, 30, 32, 21, 21]

list_classes_address = []
for i in list_classes:
    address_index = [x for x in range(len(list_classes)) if list_classes[x] == i]
    list_classes_address.append([i, address_index])
dict_address = dict(list_classes_address)
print(dict_address)
list_feature_address = []
list_feature_avg = []
for i, v in dict_address.items():
    feature = 0
    for j in v:
        feature += int(list_features[j])
    list_feature_address.append(i)
    list_feature_avg.append(round(feature / int(len(v)), 2))
print(list_feature_address)
print(list_feature_avg)