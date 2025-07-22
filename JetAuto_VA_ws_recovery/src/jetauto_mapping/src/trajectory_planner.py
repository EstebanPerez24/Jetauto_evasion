import cv2
import numpy as np
import heapq
import rospy
from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Path

def load_map(map_path):
    """Carga la imagen del mapa y la convierte a binario."""
    mapa = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    _, mapa_binario = cv2.threshold(mapa, 200, 255, cv2.THRESH_BINARY)
    return mapa, mapa_binario

def astar(mapa, start, goal):
    """Algoritmo A* para encontrar la trayectoria óptima."""
    h, w = mapa.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: np.linalg.norm(np.array(start) - np.array(goal))}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < w and 0 <= neighbor[1] < h and mapa[neighbor[1], neighbor[0]] == 255:
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + np.linalg.norm(np.array(neighbor) - np.array(goal))
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return []

def convert_to_real_coords(path, resolution, origin):
    """Convierte coordenadas de píxeles a metros usando la resolución y origen del mapa."""
    return [(x * resolution + origin[0], y * resolution + origin[1]) for (x, y) in path]

def plot_path(mapa, path):
    """Dibuja la trayectoria sobre la imagen del mapa y espera hasta que se cierre la ventana."""
    for point in path:
        cv2.circle(mapa, point, 1, (0, 0, 255), -1)

    window_name = "Trayectoria Generada"
    cv2.imshow(window_name, mapa)

    try:
        while not rospy.is_shutdown():
            key = cv2.waitKey(100)
            # Si se cierra la ventana manualmente, salimos
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()

def publish_path(path_real):
    """Publica la trayectoria en ROS como una secuencia de objetivos y la muestra en RViz."""
    rospy.init_node("path_publisher", anonymous=True)
    pub_goal = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
    pub_path = rospy.Publisher("/planned_path", Path, queue_size=10)
    rospy.sleep(1)

    path_msg = Path()
    path_msg.header.frame_id = "map"

    for point in path_real:
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        pose_stamped.pose.position.x = point[0]
        pose_stamped.pose.position.y = point[1]
        pose_stamped.pose.orientation.w = 1.0
        path_msg.poses.append(pose_stamped)
    pub_path.publish(path_msg)
    rospy.sleep(1)

    for point in path_real:
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        pose_stamped.pose.position.x = point[0]
        pose_stamped.pose.position.y = point[1]
        pose_stamped.pose.orientation.w = 1.0
        pub_goal.publish(pose_stamped)
        rospy.sleep(1)

def main():
    map_path = "/home/steven/prueba_ws/src/jetauto_codigo/jetauto_mapping/maps/mi_mapa.pgm"
    resolution = 0.01  # Extraído de map.yaml
    origin = (0.0, 0.0)  # Extraído de map.yaml

    mapa, mapa_binario = load_map(map_path)
    start = (0, 0)   # Definir manualmente o dinámicamente
    goal = (900, 650)  # Definir manualmente o dinámicamente

    path = astar(mapa_binario, start, goal)
    if not path:
        print("No se encontró un camino válido")
        return

    plot_path(mapa, path)

    path_real = convert_to_real_coords(path, resolution, origin)
    print("Trayectoria en coordenadas reales:", path_real)

    publish_path(path_real)

    # Espera antes de finalizar para asegurar publicación
    rospy.sleep(2)

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        print("Nodo interrumpido por el usuario.")
