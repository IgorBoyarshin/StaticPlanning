import arcade as ARC


SCREEN_WIDTH = 700
SCREEN_HEIGHT = 1000

CORE_WIDTH = 60
TICK_WIDTH = CORE_WIDTH
TICK_HEIGHT = 25


class Task:
    def __init__(self, core=0, start=0, length=0, index=0):
        self.core = core
        self.start = start
        self.length = length
        self.index = index

    def bottom_left(self):
        HEIGHT = self.length * TICK_HEIGHT
        START = CORE_WIDTH + (self.start-1) * TICK_HEIGHT
        x = self.core * CORE_WIDTH
        y = SCREEN_HEIGHT - (START + HEIGHT)
        return x,y

    def upper_right(self):
        WIDTH = TICK_WIDTH
        START = CORE_WIDTH + (self.start-1) * TICK_HEIGHT
        x = self.core * CORE_WIDTH + WIDTH
        y = SCREEN_HEIGHT - START
        return x,y



class Link:
    def __init__(self, src_core=0, dst_core=0, start=0, weight=0, src_task=0, dst_task=0):
        self.src_core = src_core
        self.dst_core = dst_core
        self.weight = weight
        self.start = start
        self.src_task = src_task
        self.dst_task = dst_task


def draw_field(cores, max_tick):
    LAST_COORD = max_tick * TICK_HEIGHT + CORE_WIDTH
    for core in range(0, cores + 2):
        ARC.draw_line(
                core * CORE_WIDTH,
                SCREEN_HEIGHT,
                core * CORE_WIDTH,
                SCREEN_HEIGHT - LAST_COORD,
                ARC.color.BLACK, 1)
        if core > 0 and core <= cores:
            ARC.draw_text(str(core),
                    core * CORE_WIDTH + CORE_WIDTH/3,
                    SCREEN_HEIGHT - CORE_WIDTH/4,
                    ARC.color.BLACK, 12, anchor_x="left", anchor_y="top")
    for tick in range(0, max_tick + 1):
        ARC.draw_line(
                0,
                SCREEN_HEIGHT - tick * TICK_HEIGHT - CORE_WIDTH,
                CORE_WIDTH * (cores + 1),
                SCREEN_HEIGHT - tick * TICK_HEIGHT - CORE_WIDTH,
                ARC.color.BLACK, 1)
        if tick > 0:
            ARC.draw_text(str(tick),
                    CORE_WIDTH / 4,
                    SCREEN_HEIGHT - (tick-1) * TICK_HEIGHT - CORE_WIDTH/1,
                    ARC.color.BLACK, 12, anchor_x="left", anchor_y="top")


# 0-based indexing
def draw_task(task):
    x1,y1 = task.bottom_left()
    x2,y2 = task.upper_right()
    w,h = x2-x1, y2-y1
    ARC.draw_xywh_rectangle_filled(x1, y1, w, h, ARC.color.HELIOTROPE_GRAY)
    ARC.draw_xywh_rectangle_outline(x1, y1, w, h, ARC.color.BLACK, 4)
    ARC.draw_text(str(task.index + 1),
            x1 + CORE_WIDTH/2,
            y1 + (int(task.length / 2)+1) * TICK_HEIGHT,
            ARC.color.BLACK, 12, anchor_x="left", anchor_y="bottom")


def draw_line(line, tasks):
    x1 = CORE_WIDTH * line.src_core + CORE_WIDTH/2
    y1 = SCREEN_HEIGHT - (CORE_WIDTH + (line.start-1) * TICK_HEIGHT)
    x2 = CORE_WIDTH * line.dst_core + CORE_WIDTH/2
    y2 = SCREEN_HEIGHT - (CORE_WIDTH + (line.start+line.weight-1) * TICK_HEIGHT)
    ARC.draw_line(x1,y1,x2,y2, ARC.color.RED, 4)
    ARC.draw_text(str(line.src_task+1) + "-" + str(line.dst_task+1), (x1+x2)/2, (y1+y2)/2, ARC.color.GREEN, 14)
    ARC.draw_point(x1, y1, ARC.color.BLUE, 6)
    ARC.draw_point(x2, y2, ARC.color.BLUE, 6)


def amount_cores_ticks(tasks):
    max_core = 1
    max_tick = 1
    for task in tasks:
        if task.core > max_core:
            max_core = task.core
        finish = task.start + task.length
        if finish > max_tick:
            max_tick = finish
    return max_core,(max_tick-1)


def draw(tasks, connections):
    cores,ticks = amount_cores_ticks(tasks)
    draw_field(cores, ticks)
    for task in tasks:
        draw_task(task)
    for line in connections:
        draw_line(line, tasks)


def read_from_file(path):
    in_task = False
    in_link = False
    current_object = None
    tasks = []
    links = []
    for line in [line.rstrip('\n') for line in open(path)]:
        if line == "OutTask":
            if in_task:
                tasks.append(current_object)
            elif in_link:
                links.append(current_object)
            in_task = True
            in_link = False
            current_object = Task()
        elif line == "OutLink":
            if in_task:
                tasks.append(current_object)
            elif in_link:
                links.append(current_object)
            in_task = False
            in_link = True
            current_object = Link()
        else:
            parts = line.split(':')
            field = parts[0]
            data  = parts[1]
            if in_task:
                if field == "weight":
                    current_object.length = int(data)
                elif field == "start":
                    current_object.start = int(data)
                elif field == "proc":
                    current_object.core = int(data)
                elif field == "id":
                    current_object.index = int(data)
            elif in_link:
                if field == "src_core":
                    current_object.src_core = int(data)
                elif field == "dst_core":
                    current_object.dst_core = int(data)
                elif field == "weight":
                    current_object.weight = int(data)
                elif field == "start":
                    current_object.start = int(data)
                elif field == "src_task":
                    current_object.src_task = int(data)
                elif field == "dst_task":
                    current_object.dst_task = int(data)
    # The final one
    if in_task:
        tasks.append(current_object)
    elif in_link:
        links.append(current_object)

    return tasks,links



def main():
    tasks,links = read_from_file("planning.txt")
    ARC.open_window(SCREEN_WIDTH, SCREEN_HEIGHT, "Static Planning")
    ARC.set_background_color(ARC.color.BATTLESHIP_GREY)
    ARC.start_render()
    draw(tasks, links)
    ARC.finish_render()
    ARC.run()


if __name__ == "__main__":
    main()
