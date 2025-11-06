from enviroment.entity import AdvancedContainerEntity, ContainerEntity, Entity
from enviroment.position import Position

class Table(ContainerEntity):
    """Generic table entity."""
    def __init__(self, pos: Position | None = None, material: str | None = None, description: str | None = None):
        super().__init__(
            name="table",
            pos=pos,
            material=material,
            description=description,
            uniqueness=0.6,
            prominence=0.8,
            visibility=1.0,
        )


class Shelf(ContainerEntity):
    """Generic shelf with open compartments."""
    def __init__(self, pos: Position | None = None, material: str | None = None, description: str | None = None):
        super().__init__(
            name="shelf",
            pos=pos,
            material=material,
            description=description,
            uniqueness=0.5,
            prominence=0.9,
            visibility=0.9,
        )


class Chest(AdvancedContainerEntity):
    """Generic storage chest."""
    def __init__(self, pos: Position | None = None, material: str | None = None, description: str | None = None):
        super().__init__(
            name="chest",
            pos=pos,
            material=material,
            description=description,
            uniqueness=0.5,
            prominence=0.8,
            is_open=False,
            is_locked=False,
            visibility=0.8,
            visibility_closed=0.0,
        )

class Box(ContainerEntity):
    """Generic storage box."""
    def __init__(self, pos: Position | None = None, material: str | None = None, description: str | None = None):
        super().__init__(
            name="chest",
            pos=pos,
            material=material,
            description=description,
            uniqueness=0.5,
            prominence=0.8,
            visibility=0.7,
        )



class Cabinet(AdvancedContainerEntity):
    """Generic cabinet with doors."""
    def __init__(self, pos: Position | None = None, material: str | None = None, description: str | None = None):
        super().__init__(
            name="cabinet",
            pos=pos,
            material=material,
            description=description,
            uniqueness=0.6,
            prominence=0.9,
            is_open=False,
            is_locked=False,
            visibility=0.9,
            visibility_closed=0.0,
        )


# --- TABLES ---

class WoodenTable(Table):
    """A heavy wooden table with timeworn surface."""
    def __init__(self, pos: Position | None = None):
        super().__init__(
            pos=pos,
            material="oak wood",
            description="A sturdy oak table with carved legs and a few burn marks from candles."
        )


class IronTable(Table):
    """A cold, industrial table of iron."""
    def __init__(self, pos: Position | None = None):
        super().__init__(
            pos=pos,
            material="forged iron",
            description="An iron table with riveted joints, cold to the touch and faintly humming with resonance."
        )


# --- SHELVES ---

class WoodenShelf(Shelf):
    """A shelf made of old planks, uneven but reliable."""
    def __init__(self, pos: Position | None = None):
        super().__init__(
            pos=pos,
            material="pine wood",
            description="An open shelf of rough pine planks, filled with dust and faint smell of resin."
        )


class MetalShelf(Shelf):
    """A shelf made of steel, functional and modern."""
    def __init__(self, pos: Position | None = None):
        super().__init__(
            pos=pos,
            material="steel",
            description="A steel shelf with industrial precision, polished surfaces reflecting dim light."
        )


# --- CHESTS ---

class PaperBox(Box):
    """A lightweight cardboard box, easy to open and inspect."""
    def __init__(self, pos: Position | None = None):
        super().__init__(
            pos=pos,
            material="cardboard",
            description=(
                "A plain cardboard box with slightly frayed edges. "
                "The lid bends easily when pressed, and it carries faint marks from tape and labels."
            ),
        )
        # It's typically not locked and opens without resistance
        self.is_open = False
        self.is_locked = False

        # Visibility: contents somewhat perceivable even when closed
        self.visibility_open = 1.0
        self.visibility_closed = 0.3

        # Slightly more generic and less prominent visually than a wooden chest
        self.uniqueness = 0.4
        self.prominence = 0.7


class WoodenChest(Chest):
    """Classic wooden storage chest."""
    def __init__(self, pos: Position | None = None):
        super().__init__(
            pos=pos,
            material="oak wood reinforced with iron bands",
            description="An old chest with darkened oak panels and a heavy iron lockplate."
        )


class IronChest(Chest):
    """Reinforced iron chest for valuables."""
    def __init__(self, pos: Position | None = None):
        super().__init__(
            pos=pos,
            material="forged iron",
            description="A compact chest of black iron, its hinges groaning faintly when moved."
        )


# --- CABINETS ---

class WoodenCabinet(Cabinet):
    """Tall wooden cabinet with panel doors."""
    def __init__(self, pos: Position | None = None):
        super().__init__(
            pos=pos,
            material="varnished walnut",
            description="A tall walnut cabinet with two panel doors, faint scent of polish and dust within."
        )


class IronCabinet(Cabinet):
    """Industrial cabinet of riveted iron plates."""
    def __init__(self, pos: Position | None = None):
        super().__init__(
            pos=pos,
            material="riveted iron",
            description="A heavy iron cabinet, painted grey, its surface chipped and marked by years of use."
        )
