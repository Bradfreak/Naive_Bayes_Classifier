def customArgmax(data):
    maxKey = None
    maxValue = None

    for key in data:
        if maxKey == None:
            maxKey = key
        if maxValue == None or maxValue < data[key]:
            maxValue = data[key]
            maxKey = key

    return maxKey