"""Word and sentence lists transcribed from the paper
"Emergent Introspective Awareness in Large Language Models" (appendix 12.1.1,
12.2.2 and the intentional-control prompt excerpt). Extracted programmatically
from the PDF; do not edit. Notes:
- The paper's baseline list contains "Butterflies" twice; we deduplicated it
  (99 unique words) per explicit user decision 2026-07-08.
- Curly apostrophes from PDF typesetting were normalized to ASCII "'".
- Words are stored capitalized as in the paper; prompts must lowercase them
  ("the word is always written in lowercase").
"""

# 100 baseline words: activations averaged over these are subtracted to form
# every concept vector.
BASELINE_WORDS_PAPER: list[str] = [
    "Desks", "Jackets", "Gondolas", "Laughter", "Intelligence", "Bicycles",
    "Chairs", "Orchestras", "Sand", "Pottery", "Arrowheads", "Jewelry",
    "Daffodils", "Plateaus", "Estuaries", "Quilts", "Moments", "Bamboo",
    "Ravines", "Archives", "Hieroglyphs", "Stars", "Clay", "Fossils",
    "Wildlife", "Flour", "Traffic", "Bubbles", "Honey", "Geodes",
    "Magnets", "Ribbons", "Zigzags", "Puzzles", "Tornadoes", "Anthills",
    "Galaxies", "Poverty", "Diamonds", "Universes", "Vinegar", "Nebulae",
    "Knowledge", "Marble", "Fog", "Rivers", "Scrolls", "Silhouettes",
    "Marbles", "Cakes", "Valleys", "Whispers", "Pendulums", "Towers",
    "Tables", "Glaciers", "Whirlpools", "Jungles", "Wool", "Anger",
    "Ramparts", "Flowers", "Research", "Hammers", "Clouds", "Justice",
    "Dogs", "Butterflies", "Needles", "Fortresses", "Bonfires", "Skyscrapers",
    "Caravans", "Patience", "Bacon", "Velocities", "Smoke", "Electricity",
    "Sunsets", "Anchors", "Parchments", "Courage", "Statues", "Oxygen",
    "Time", "Fabric", "Pasta", "Snowflakes", "Mountains",
    "Echoes", "Pianos", "Sanctuaries", "Abysses", "Air", "Dewdrops",
    "Gardens", "Literature", "Rice", "Enigmas",
]

# 50 words for which concept vectors are computed and which fill the {word}
# slot of the think / don't-think prompts.
CONCEPT_WORDS_PAPER: list[str] = [
    "Dust", "Satellites", "Trumpets", "Origami", "Illusions", "Cameras",
    "Lightning", "Constellations", "Treasures", "Phones", "Trees", "Avalanches",
    "Mirrors", "Fountains", "Quarries", "Sadness", "Xylophones", "Secrecy",
    "Oceans", "Information", "Deserts", "Kaleidoscopes", "Sugar", "Vegetables",
    "Poetry", "Aquariums", "Bags", "Peace", "Caverns", "Memories",
    "Frosts", "Volcanoes", "Boulders", "Harmonies", "Masquerades", "Rubber",
    "Plastic", "Blood", "Amphitheaters", "Contraptions", "Youths", "Dynasties",
    "Snow", "Dirigibles", "Algorithms", "Denim", "Monoliths", "Milk",
    "Bread", "Silver",
]

# 100 control words: their concept vectors give the null/"shaded region"
# baseline against which the target word's cosine is compared.
CONTROL_WORDS_PAPER: list[str] = [
    "Islands", "Observatories", "Ice", "Darkness", "Computers", "Children",
    "Forests", "Linen", "Trains", "Software", "Happiness", "Salt",
    "Mechanisms", "Thunder", "Lagoons", "Carousels", "Advice", "Pepper",
    "Ghosts", "Fireworks", "Crystals", "Blueprints", "Wisdom", "Embers",
    "Cotton", "Strawberries", "Elephants", "Zebras", "Gasoline", "Horizons",
    "Periscopes", "Glitters", "Dreams", "Thunders", "Love", "Candles",
    "Coronets", "Houses", "Vegetation", "Beef", "Tea", "Whirlwinds",
    "Bridges", "Mud", "Cups", "Telescopes", "Sunshine", "Zeppelins",
    "Seafood", "Monorails", "Jewels", "Footwear", "Copper", "Education",
    "Beer", "Journeys", "Kittens", "Granite", "Oases", "Timber",
    "Villages", "Spectacles", "Compasses", "Glue", "Cathedrals", "Rockets",
    "Handprints", "Baskets", "Shadows", "Meadows", "Ladders", "Steam",
    "Buildings", "Symphonies", "Geysers", "Porcelain", "Livestock", "Mail",
    "Freedom", "Cutlery", "Inkwells", "Foam", "Shipwrecks", "Equipment",
    "Horses", "Mazes", "Chaos", "Umbrellas", "Catapults", "Scarves",
    "Pillows", "Windmills", "Windows", "Music", "Machinery", "Kingdoms",
    "Gargoyles", "Questions", "Books", "Relics",
]

# 50 sentences for the {sentence} slot.
SENTENCES_PAPER: list[str] = [
    "The old clock on the wall ticked loudly.",
    "She collected seashells every summer at the beach.",
    "The cat jumped onto the windowsill to watch birds.",
    "His favorite ice cream flavor was mint chocolate chip.",
    "The book fell open to page 217.",
    "Lightning flashed across the night sky.",
    "They planted tulip bulbs in the garden last fall.",
    "The coffee shop was bustling with morning customers.",
    "She tied her hiking boots with double knots.",
    "The museum exhibit featured ancient Egyptian artifacts.",
    "Children laughed as they ran through the sprinkler.",
    "The train arrived precisely on schedule.",
    "He couldn't remember where he had parked his car.",
    "Autumn leaves crunched beneath their feet.",
    "The recipe called for two teaspoons of vanilla extract.",
    "The dog wagged its tail excitedly at the park.",
    "Mountains loomed in the distance, covered with snow.",
    "She practiced piano for three hours every day.",
    "The telescope revealed stunning details of Saturn's rings.",
    "Fresh bread was baking in the oven.",
    "They watched the sunset from the rooftop.",
    "The professor explained the theory with great enthusiasm.",
    "Waves crashed against the rocky shoreline.",
    "He assembled the furniture without reading the instructions.",
    "Stars twinkled brightly in the clear night sky.",
    "The old photograph brought back forgotten memories.",
    "Bees buzzed around the flowering cherry tree.",
    "She solved the crossword puzzle in record time.",
    "The air conditioner hummed quietly in the background.",
    "Rain pattered softly against the windowpane.",
    "The movie theater was packed for the premiere.",
    "He sketched the landscape with charcoal pencils.",
    "Children built sandcastles at the water's edge.",
    "The orchestra tuned their instruments before the concert.",
    "Fragrant lilacs bloomed along the garden fence.",
    "The basketball bounced off the rim.",
    "She wrapped the birthday present with blue ribbon.",
    "The hiker followed the trail markers through the forest.",
    "Their canoe glided silently across the still lake.",
    "The antique vase was carefully wrapped in bubble wrap.",
    "Fireflies flickered in the summer twilight.",
    "The chef garnished the plate with fresh herbs.",
    "Wind chimes tinkled melodically on the porch.",
    "The flight attendant demonstrated safety procedures.",
    "He repaired the leaky faucet with a new washer.",
    "Fog shrouded the valley below the mountain.",
    "The comedian's joke made everyone laugh.",
    "She planted herbs in pots on the kitchen windowsill.",
    "The painting hung crookedly on the wall.",
    "Snowflakes drifted lazily from the gray sky.",
]

assert len(BASELINE_WORDS_PAPER) == 99
assert len(CONCEPT_WORDS_PAPER) == 50
assert len(CONTROL_WORDS_PAPER) == 100
assert len(SENTENCES_PAPER) == 50
