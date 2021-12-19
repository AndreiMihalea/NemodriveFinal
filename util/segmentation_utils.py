def compute_miou(overlay, predicted):
    predicted = predicted / predicted.max()
    c1 = overlay.astype(bool)
    c2 = predicted.astype(bool)

    overlap = c1 * c2
    union = c1 + c2

    miou = overlap.sum() / float(union.sum())

    return miou


def compute_score(overlay, predicted):
    return predicted[overlay != 0].mean()