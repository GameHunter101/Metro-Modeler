#let Foundation = math.op("Foundation")
#let BuildingShape = math.op("BuildingShape")
#let rand = math.op("rand")
#let TierBottom = math.op("TierBottom")
#let TierTop = math.op("TierTop")
#let Roof = math.op("Roof")
#let Ledge = math.op("Ledge")
#let Storey = math.op("Storey")
#let lerp = math.op("lerp")

LOD 2
#line(length: 100%)

+ $"Footprint" -> Foundation("shape", 3) display(sum_(i=1)^n BuildingShape("pos", "shape", rand(h_min, h_max), [1, 1]))$

LOD 1
#line(length: 100%)

+ $BuildingShape("pos", "shape", h, "scale"_(x z))\ quad arrow.r.curve TierBottom("pos", "shape", 3, h, "transition", "scale"_(x z)) TierTop("pos", "shape", "scale"_(x z))$

+ $TierTop("pos", "shape", h, "scale"_(x z))\ arrow.r.curve display(cases(Roof("pos", "shape", h, "scale"_(x z)) &quad 50%, Ledge("pos", "shape", h, "scale"_(x z)) TierBottom("pos", "shape", h, h' = rand(h, h_max)) &quad 50%\ quad TierTop("pos", "shape", h', alpha dot "scale"_(x z))))$

+ $angle.l TierBottom("pos", "shape", "bot", h, underline("smooth"), "scale"_(x z)), Ledge("pos", "shape", h, "scale"_(x z)), Sigma^* angle.r -> lambda$

+ $angle.l TierBottom("pos", "shape", "bot", h, underline("smooth"), "scale"_(x z)), TierBottom("pos", "shape", h, h', "transition", "scale"'_(x z)), Sigma^* angle.r\ quad arrow.r.curve TierBottom("pos", "shape", h, h', "transition", ["scale"_(x z)[0], "scale"'_(x z)[1]])$

LOD 0
#line(length: 100%)

+ $TierBottom("pos", "shape", "bot", h, "scale"_(x z))\ quad arrow.r.curve display(sum_(i=1)^(h  -"bot" - 1)) Storey("pos", "shape", i, "core", [lerp("scale"_(x z), i / (h - "bot")), lerp("scale"_(x z), (i + 1) / (h - "bot"))])$

+ $Foundation("shape", h) -> Storey("pos", "shape", i, "foundation", [1, 1])$
