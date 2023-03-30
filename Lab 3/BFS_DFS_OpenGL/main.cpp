#include <iostream>
#include <random>
#include <vector>
#include <queue>
#include <stack>
#include <unistd.h>
#include <GL/glut.h>

using namespace std;

int random(int min, int max){
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(min, max);
    return dis(gen);
}

// Estructura de datos para un nodo
struct Node {
    int id;             // identificador del nodo
    vector<Node*> adj;  // lista de adyacencia para almacenar los nodos adyacentes

    bool visited;       // variable para saber si el nodo ya fue visitado
};

// Estructura de datos para una arista
struct Edge {
    Node* from;  // nodo de inicio de la arista
    Node* to;    // nodo final de la arista
};

int n = 99; // tamaño del grafo (4x4 en este caso)
vector<Node*> nodes(n*n);  // vector de nodos

// Función para crear un grafo de nxn
vector<Node*> createGraph(int n) {

    // Crear todos los nodos y asignar un identificador único
    for (int i = 0; i < n*n; i++) {
        nodes[i] = new Node;
        nodes[i]->id = i;
    }

    // Agregar aristas horizontales y verticales
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int index = i*n + j;  // índice del nodo actual

            // Agregar arista horizontal
            if (j < n-1) {
                Edge e;
                e.from = nodes[index];
                e.to = nodes[index+1];
                nodes[index]->adj.push_back(nodes[index+1]);
            }

            // Agregar arista vertical
            if (i < n-1) {
                Edge e;
                e.from = nodes[index];
                e.to = nodes[index+n];
                nodes[index]->adj.push_back(nodes[index+n]);
            }
        }
    }

    //crear aristas diagonales
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int index = i*n + j;  // índice del nodo actual

            // Agregar arista diagonal 1
            if (i < n-1 && j < n-1) {
                Edge e;
                e.from = nodes[index];
                e.to = nodes[index+n+1];
                nodes[index]->adj.push_back(nodes[index+n+1]);
            }

            // Agregar arista diagonal 2
            if (i < n-1 && j > 0) {
                Edge e;
                e.from = nodes[index];
                e.to = nodes[index+n-1];
                nodes[index]->adj.push_back(nodes[index+n-1]);
            }
        }
    }

    return nodes;
}

// Función para eliminar un porcentaje aleatorio de nodos con sus respectivas aristas
void removeRandomNodes(double percentage) {
    // Crear un generador de números aleatorios
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, n*n-1);

    // Eliminar un porcentaje de nodos
    int numNodesToRemove = (int) (percentage * n*n);
    for (int i = 0; i < numNodesToRemove; i++) {
        int index = dis(gen);  // índice del nodo a eliminar
        nodes[index] = NULL;
    }

    // Eliminar las aristas que apuntan a los nodos eliminados
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int index = i*n + j;  // índice del nodo actual
            if (nodes[index] != nullptr) {
                for (int k = 0; k < nodes[index]->adj.size(); k++) {
                    Node* adj = nodes[index]->adj[k];
                    if (adj == nullptr) {
                        nodes[index]->adj.erase(nodes[index]->adj.begin() + k);
                    }
                }
            }
        }
    }

    // Eliminar las aristas que salen de los nodos eliminados
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int index = i*n + j;  // índice del nodo actual
            if (nodes[index] == nullptr) {
                for (int k = 0; k < n; k++) {
                    for (int l = 0; l < n; l++) {
                        int index2 = k*n + l;  // índice del nodo actual
                        if (nodes[index2] != nullptr) {
                            for (int m = 0; m < nodes[index2]->adj.size(); m++) {
                                Node* adj = nodes[index2]->adj[m];
                                if (adj != nullptr && adj->id == index) {
                                    nodes[index2]->adj.erase(nodes[index2]->adj.begin() + m);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Función para dibujar el grafo
void drawGraph() {
    /* // Dibujar los nodos
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int index = i*n + j;  // índice del nodo actual
            if (nodes[index] != NULL) {
                glPushMatrix();
                glTranslatef(j, n-i, 0);
                glutSolidCube(0.5);
                glPopMatrix();
            }
        }
    }

    // Dibujar las aristas
    glColor3f(0, 0, 0);
    glLineWidth(2);
    glBegin(GL_LINES);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int index = i*n + j;  // índice del nodo actual
            if (nodes[index] != NULL) {
                for (int k = 0; k < nodes[index]->adj.size(); k++) {
                    Node* adj = nodes[index]->adj[k];
                    if (adj != NULL) {
                        glVertex2f(j, n-i);
                        glVertex2f(adj->id % n, n - adj->id / n);
                    }
                }
            }
        }
    }
    glEnd(); */
    // Dibujar los nodos
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int index = i*n + j;  // índice del nodo actual
            if (nodes[index] != NULL) {
                glPushMatrix();
                glTranslatef(j, n-i, 0);
                if (nodes[index]->visited) {
                    glColor3f(0, 1, 0); // pintar el nodo de verde si fue visitado
                } else {
                    glColor3f(1, 1, 1); // pintar el nodo de blanco si no fue visitado
                }
                glutSolidCube(0.5);
                glPopMatrix();
            }
        }
    }

    // Dibujar las aristas
    glColor3f(0, 0, 0);
    glLineWidth(2);
    glBegin(GL_LINES);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int index = i*n + j;  // índice del nodo actual
            if (nodes[index] != NULL) {
                for (int k = 0; k < nodes[index]->adj.size(); k++) {
                    Node* adj = nodes[index]->adj[k];
                    if (adj != NULL) {
                        glVertex2f(j, n-i);
                        glVertex2f(adj->id % n, n - adj->id / n);
                    }
                }
            }
        }
    }
    glEnd();
}

/////////////////////////////////////////////////////
// Función BFS
void bfs(Node* start, Node* end) {// Crear una cola
    queue<Node*> q;

    // Marcar todos los nodos como no visitados
    for (int i = 0; i < n*n; i++) {
        if (nodes[i] != nullptr) {
            nodes[i]->visited = false;
        }
    }

    // Marcar el nodo inicial como visitado y agregarlo a la cola
    start->visited = true;
    q.push(start);

    // Mientras la cola no esté vacía
    while (!q.empty()) {
        // Sacar el primer elemento de la cola
        Node* u = q.front();
        q.pop();

        // Si el nodo actual es el nodo final, terminar
        if (u == end) {
            break;
        }

        // Recorrer los nodos adyacentes
        for (int i = 0; i < u->adj.size(); i++) {
            Node* v = u->adj[i];
            if (v != nullptr && !v->visited) {
                v->visited = true;
                q.push(v);

                // Pintar la arista que lleva al nodo visitado,si es que existe una arista entre los dos nodos (no es un nodo eliminado)
                if (u->adj[i] != NULL) {
                    /*glBegin(GL_LINES);
                    glColor3f(1, 0, 0);
                    glVertex2f(node->id % n, n - node->id / n);
                    glVertex2f(adj->id % n, n - adj->id / n);
                    glEnd();*/
                    glColor3f(1, 0, 0);
                    glLineWidth(2);
                    glBegin(GL_LINES);
                    glVertex2f(u->id % n, n - u->id / n);
                    glVertex2f(v->id % n, n - v->id / n);
                    glEnd();
                }

                // Esperar un poco para poder visualizar la animación
                usleep(100000);
                glutSwapBuffers();
            }
        }
    }
}

// Función DFS
void dfs(Node* start, Node* end) {
    // Reiniciar la variable visited de cada nodo
    for (int i = 0; i < n*n; i++) {
        if (nodes[i] != nullptr) {
            nodes[i]->visited = false;
        }
    }

    // Crear una pila para el DFS
    stack<Node*> s;

    // Agregar el nodo inicial a la pila
    s.push(start);
    start->visited = true;

    // Realizar el DFS
    while (!s.empty()) {
        // Sacar un nodo de la pila
        Node* node = s.top();
        s.pop();

        // Visitar los nodos adyacentes que no han sido visitados
        for (int i = 0; i < node->adj.size(); i++) {
            Node* adj = node->adj[i];
            if (adj != nullptr && !adj->visited) {
                adj->visited = true;
                s.push(adj);

                // Pintar la arista que lleva al nodo visitado,si es que existe una arista entre los dos nodos (no es un nodo eliminado)
                if (node->adj[i] != nullptr) {
                    /*glBegin(GL_LINES);
                    glColor3f(0, 0, 1);
                    glVertex2f(node->id % n, n - node->id / n);
                    glVertex2f(adj->id % n, n - adj->id / n);
                    glEnd();*/
                    glColor3f(0, 0, 1);
                    glLineWidth(2);
                    glBegin(GL_LINES);
                    glVertex2f(node->id % n, n - node->id / n);
                    glVertex2f(adj->id % n, n - adj->id / n);
                    glEnd();
                }

                // Esperar un poco para poder visualizar la animación
                usleep(100000);
                glutSwapBuffers();
            }
        }
    }
}
////////////////////////////////////////////////////

// Función para redimensionar la ventana
void reshape(int w, int h) {
    glViewport(0, 0, w, h);
}

// funcion para hacer zoom con el mouse
void mouseWheel(int button, int dir, int x, int y) {
    if (button == 3) {
        glScalef(1.1, 1.1, 1.1);
    }
    if (button == 4) {
        glScalef(0.9, 0.9, 0.9);
    }
    glutPostRedisplay();
}

//funcion para elegir en tiempo de ejecucion el algoritmo a utilizar
void keyboard(unsigned char key, int x, int y) {
    if (key == 'b') {
        //Node* start1 = nodes[0];
        //Node* start1 = nodes[random(5, n*n-1)];
        Node* start1 = nodes[n*n / 2]; // centro del grafo
        Node* end1 = nodes[n*n - 1];
        bfs(start1, end1);
    } else if (key == 'd') {
        //Node* start = nodes[0];
        //Node* start = nodes[random(5, n*n-1)];
        Node* start = nodes[n*n / 2]; // centro del grafo
        Node* end = nodes[n*n - 1];
        dfs(start, end);
    }else if (key == 'r') {
        // Reiniciar el grafo
        for (int i = 0; i < n*n; i++) {
            if (nodes[i] != nullptr) {
                nodes[i]->visited = false;
            }
        }
        glutPostRedisplay();
    }
}

// Función init
void init() {
    glClearColor(1, 1, 1, 1);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, n, 0, n, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);

    // Dibujar el grafo
    drawGraph();

    // Llamar a la función BFS desde el nodo 0
    /*Node* start1 = nodes[0];
   //Node* start1 = nodes[random(5, n*n-1)];
   Node* end1 = nodes[n*n - 1];
   bfs(start1, end1); // */

    // Llamar a la función DFS desde el nodo 0
    /*Node* start = nodes[0];
    //Node* start = nodes[random(5, n*n-1)];
    Node* end = nodes[n*n-1];
    dfs(start, end); // */


    glutSwapBuffers();
}

int main(int argc, char** argv) {
    // Crear el grafo
    nodes = createGraph(n);

    // Eliminar un porcentaje aleatorio de nodos
    removeRandomNodes(0.3);

    // Inicializar la ventana
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(600, 600);
    glutCreateWindow("Grafo");
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouseWheel);
    glutKeyboardFunc(keyboard);
    init();
    glutMainLoop();

    return 0;
}