"""The ten OpenStreetMap "folk tags" Day 21 maps. Shared by extract.py and build.py.

Each entry: key (used in data + page), label (and `one`, its singular), the OSM tag it matches (shown on the page, so readers see
the raw tag), a test on the feature's tags, and one line on what the thing is.
"""


def _has_value(key, value):
    # a tag can hold several values separated by ";" (e.g. "ALPR;speed_camera")
    return lambda t: value in (t.get(key) or "").split(";")


FOLK = [
    {"key": "ghost_bike", "label": "Ghost bikes", "one": "Ghost bike", "tag": "memorial=ghost_bike",
     "test": _has_value("memorial", "ghost_bike"),
     "about": "White-painted bicycles locked up where a cyclist was killed."},
    {"key": "alpr", "label": "License-plate readers", "one": "License-plate reader", "tag": "surveillance:type=ALPR",
     "test": _has_value("surveillance:type", "ALPR"),
     "about": "Cameras that read every passing license plate. Mappers spell their operator, NYC DOT, at least three different ways."},
    {"key": "fire_alarm_box", "label": "Fire alarm boxes", "one": "Fire alarm box", "tag": "emergency=fire_alarm_box",
     "test": _has_value("emergency", "fire_alarm_box"),
     "about": "Street-corner call boxes that ring straight to the fire department."},
    {"key": "playground_map", "label": "Painted playground maps", "one": "Painted playground map", "tag": "playground=map",
     "test": _has_value("playground", "map"),
     "about": "Maps painted on playground asphalt: the US, the world, NYC, even the solar system. Maps on a map."},
    {"key": "hopscotch", "label": "Hopscotch grids", "one": "Hopscotch grid", "tag": "playground=hopscotch",
     "test": _has_value("playground", "hopscotch"),
     "about": "Hopscotch courts painted on playgrounds and schoolyards."},
    {"key": "tree_stump", "label": "Tree stumps", "one": "Tree stump", "tag": "natural=tree_stump",
     "test": _has_value("natural", "tree_stump"),
     "about": "What's left after a street or park tree comes down."},
    {"key": "bookcase", "label": "Little free libraries", "one": "Little free library", "tag": "amenity=public_bookcase",
     "test": _has_value("amenity", "public_bookcase"),
     "about": "Take-a-book, leave-a-book boxes and shelves."},
    {"key": "memorial_bench", "label": "Memorial benches", "one": "Memorial bench", "tag": "openbenches:id",
     "test": lambda t: "openbenches:id" in t or _has_value("memorial", "bench")(t),
     "about": "Benches with a dedication plaque, most logged by the OpenBenches project."},
    {"key": "poop_bags", "label": "Poop-bag dispensers", "one": "Poop-bag dispenser", "tag": "vending=excrement_bags",
     "test": _has_value("vending", "excrement_bags"),
     "about": "Free bags for picking up after your dog."},
    {"key": "closed_shop", "label": "Shops that closed", "one": "Closed shop", "tag": "disused:shop=*",
     "test": lambda t: "disused:shop" in t,
     "about": "Storefronts mappers marked as closed. About a third also record what the shop used to be."},
]

# tags worth showing in a tooltip, in this order (first one present becomes the bold line)
SHOW = ["name", "disused:name", "old_name", "inscription", "subject", "operator", "manufacturer",
        "description", "disused:shop", "start_date", "ref"]
