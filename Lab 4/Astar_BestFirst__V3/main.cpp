#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <SFML/Graphics.hpp>
#include <chrono>
#include <unordered_set>

constexpr float WIDTH = 700;
constexpr float HEIGHT = 700;
int SIZE = 30;
sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "A* Pathfinder", sf::Style::Close);
sf::Event evnt;
sf::Font font;
sf::Text text;

const sf::Vector2f BlockSize = { 8, 8 };
const size_t CollumsX = WIDTH / BlockSize.x;
const size_t CollumsY = HEIGHT / BlockSize.y;

const sf::Color OutlineBlockColor = sf::Color::Black;
const sf::Color WallColor = sf::Color::Black;
const sf::Color StartColor = sf::Color::Blue;
const sf::Color EndColor = sf::Color::Red;
const sf::Color PathColor = sf::Color::Green;
const sf::Color Vertex = sf::Color::Yellow;

//random function
int random(int min, int max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(min, max);
    return dis(gen);
}

class Node {
public:
    int id;
    float x, y;
    float weight;
    float heuristic;
    std::vector<Node*> neighbors;
    bool visited;
    bool obstacle;

    //variables for A* algorithm
    float g;
    float f;
    Node* parent;

    //variables for Best First Search
    float h;

    Node(int _id, float _x, float _y, float _weight, float _heuristic) : id(_id), x(_x), y(_y), weight(_weight), heuristic(_heuristic) {}

    void addNeighbor(Node* neighbor) {
        neighbors.push_back(neighbor);
    }

    float distanceTo(Node* other) {
        float dx = x - other->x;
        float dy = y - other->y;
        return sqrt(dx * dx + dy * dy);
    }

    bool contains(int i, int i1) {
        if (x == i && y == i1) {
            return true;
        }
        return false;
    }
};

struct CompareNodeHeuristic {
    bool operator()(const Node* a, const Node* b) {
        return a->heuristic > b->heuristic;
    }
};

class MeshNetwork {
public:
    int n;
    std::vector<Node *> nodes;
    std::vector<std::vector<Node *>> grid;

    Node* Start;
    Node* End;

    MeshNetwork(int _n) : n(_n) {
        // create nodes and grid
        int id = 0;
        float spacing = 50.f;
        float origin_x = 100.f;
        float origin_y = 100.f;
        grid.resize(n);
        for (int i = 0; i < n; i++) {
            grid[i].resize(n);
            for (int j = 0; j < n; j++) {
                float x = origin_x + j * spacing;
                float y = origin_y + i * spacing;
                float weight = float(random(100, 10000)); //random weight between 1 and 100
                float heuristic = float(random(10, 10000)); //random heuristic value for A* algorithm and Best First Search
                Node *node = new Node(id, x, y, weight, heuristic);
                nodes.push_back(node);
                grid[i][j] = node;
                id++;
            }
        }
        // add edges
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Node *node = grid[i][j];
                if (i > 0 && j > 0) {
                    node->addNeighbor(grid[i - 1][j - 1]);
                }
                if (i > 0) {
                    node->addNeighbor(grid[i - 1][j]);
                }
                if (i > 0 && j < n - 1) {
                    node->addNeighbor(grid[i - 1][j + 1]);
                }
                if (j > 0) {
                    node->addNeighbor(grid[i][j - 1]);
                }
                if (j < n - 1) {
                    node->addNeighbor(grid[i][j + 1]);
                }
                if (i < n - 1 && j > 0) {
                    node->addNeighbor(grid[i + 1][j - 1]);
                }
                if (i < n - 1) {
                    node->addNeighbor(grid[i + 1][j]);
                }
                if (i < n - 1 && j < n - 1) {
                    node->addNeighbor(grid[i + 1][j + 1]);
                }
            }
        }
    }

    void deleteNodesInRange(int x1, int y1, int x2, int y2) {
        for (int i = x1; i <= x2; i++) {
            for (int j = y1; j <= y2; j++) {
                Node *node = grid[i][j];
                // remove edges to neighbors
                for (Node *neighbor: node->neighbors) {
                    neighbor->neighbors.erase(std::remove(neighbor->neighbors.begin(), neighbor->neighbors.end(), node),
                                              neighbor->neighbors.end());
                }
                // remove node from grid
                grid[i][j] = nullptr;
                // remove node from nodes
                nodes.erase(std::remove(nodes.begin(), nodes.end(), node), nodes.end());
                // delete node
                delete node;
            }
        }
    }

    Node *getNode(int x, int y) {
        return grid[x][y];
    }

    //draw path in green
    void drawPath() {
        //paint the nodese of green and the lines between them in red
        for (Node *node: nodes) {
            if (node->visited) {
                sf::CircleShape circle(5.f);
                circle.setFillColor(PathColor);
                circle.setPosition(node->x, node->y);
                window.draw(circle);
            }
        }

        for (Node *node: nodes) {
            if (node->visited) {
                for (Node *neighbor: node->neighbors) {
                    if (neighbor->visited) {
                        sf::Vertex line[] = {
                                sf::Vertex(sf::Vector2f(node->x, node->y)),
                                sf::Vertex(sf::Vector2f(neighbor->x, neighbor->y))
                        };
                        line[0].color = Vertex;
                        line[1].color = Vertex;
                        window.draw(line, 2, sf::Lines);
                    }
                }
            }
        }


    }

    //Astar algorithm to find the shortest path between two nodes. paint the path in green
    void Astar(Node *start, Node *end) {
        //reset nodes
        for (Node *node: nodes) {
            node->visited = false;
            node->parent = nullptr;
            node->g = 0;
            node->f = 0;
        }
        //create open and closed list
        std::vector<Node *> openList;
        std::vector<Node *> closedList;
        //add start node to open list
        openList.push_back(start);
        //while open list is not empty
        while (!openList.empty()) {
            //get node with lowest f value
            Node *current = openList[0];
            for (Node *node: openList) {
                if (node->f < current->f) {
                    current = node;
                }
            }
            //remove current node from open list
            openList.erase(std::remove(openList.begin(), openList.end(), current), openList.end());
            //add current node to closed list
            closedList.push_back(current);
            //if current node is end node, path has been found
            if (current == end) {
                //create path
                std::vector<Node *> path;
                Node *temp = current;
                path.push_back(temp);
                while (temp->parent != nullptr) {
                    path.push_back(temp->parent);
                    temp = temp->parent;
                }
                //paint path
                for (Node *node: path) {
                    node->visited = true;
                }
                return;
            }
            //for each neighbor of current node
            for (Node *neighbor: current->neighbors) {
                //if neighbor is not traversable or in closed list, skip
                if (neighbor->obstacle || std::find(closedList.begin(), closedList.end(), neighbor) != closedList.end()) {
                    continue;
                }
                //calculate g score
                float tempG = current->g + current->distanceTo(neighbor);
                //if neighbor is not in open list
                if (std::find(openList.begin(), openList.end(), neighbor) == openList.end()) {
                    //add neighbor to open list
                    openList.push_back(neighbor);
                } else if (tempG >= neighbor->g) {
                    //if g score is higher, skip
                    continue;
                }
                //set g score
                neighbor->g = tempG;
                //set h score
                neighbor->h = neighbor->distanceTo(end);
                //set f score
                neighbor->f = neighbor->g + neighbor->h;
                //set parent
                neighbor->parent = current;
            }
        }
    }

    float calculateHeuristic(Node *pNode, Node *pNode1) {
        //calculate heuristic for best first search
        return pNode->distanceTo(pNode1);
    }

    void BestFisrtSearch(Node *start, Node *end) {
        // best first searh algorithm is different from A* algorithm in that it does not use the FCost to determine the next node to visit
        // instead it uses the HCost to determine the next node to visit
        // this is a very simple algorithm and is not very efficient
        // it is used to show the difference between A* and best first search
        //reset nodes
        for (Node *node: nodes) {
            node->visited = false;
            node->parent = nullptr;
            node->g = 0;
            node->f = 0;
        }
        //create open and closed list
        std::vector<Node *> openList;
        std::vector<Node *> closedList;
        //add start node to open list
        openList.push_back(start);
        //while open list is not empty

        while (!openList.empty()) {
            //get node with lowest f value
            Node *current = openList[0];
            for (Node *node: openList) {
                if (node->h < current->h) {
                    current = node;
                }
            }
            //remove current node from open list
            openList.erase(std::remove(openList.begin(), openList.end(), current), openList.end());
            //add current node to closed list
            closedList.push_back(current);
            //if current node is end node, path has been found
            if (current == end) {
                //create path
                std::vector<Node *> path;
                Node *temp = current;
                path.push_back(temp);
                while (temp->parent != nullptr) {
                    path.push_back(temp->parent);
                    temp = temp->parent;
                }
                //paint path
                for (Node *node: path) {
                    node->visited = true;
                }
                return;
            }
            //for each neighbor of current node
            for (Node *neighbor: current->neighbors) {
                //if neighbor is not traversable or in closed list, skip
                if (neighbor->obstacle || std::find(closedList.begin(), closedList.end(), neighbor) != closedList.end()) {
                    continue;
                }
                //calculate g score
                float tempG = current->g + current->distanceTo(neighbor);
                //if neighbor is not in open list
                if (std::find(openList.begin(), openList.end(), neighbor) == openList.end()) {
                    //add neighbor to open list
                    openList.push_back(neighbor);
                } else if (tempG >= neighbor->g) {
                    //if g score is higher, skip
                    continue;
                }
                //set g score
                neighbor->g = tempG;
                //set h score
                neighbor->h = neighbor->distanceTo(end);
                //set f score
                neighbor->f = neighbor->g + neighbor->h;
                //set parent
                neighbor->parent = current;
            }
        }
    }

    //draw nodes and edges in the mesh network
    void drawMesh() {
        font.loadFromFile(R"(C:\Users\win 10\Documents\CLION\pruebas\arial.ttf)");
        text.setFont(font);
        text.setCharacterSize(10);

        // draw edges
        for (Node *node: nodes) {
            for (Node *neighbor: node->neighbors) {
                sf::Vertex line[] = {
                        sf::Vertex(sf::Vector2f(node->x, node->y)),
                        sf::Vertex(sf::Vector2f(neighbor->x, neighbor->y))
                };
                window.draw(line, 2, sf::Lines);
            }
        }
        // draw nodes
        for (Node *node: nodes) {
            sf::CircleShape circle(5.f);
            circle.setFillColor(sf::Color::White);
            circle.setPosition(node->x - 5.f, node->y - 5.f);
            window.draw(circle);
            text.setString(std::to_string(node->id));
            text.setPosition(node->x + 5.f, node->y - 5.f);
            window.draw(text);
        }
    }

    // visualise the mesh network
    void visualise(){
        window.setFramerateLimit(33);
        //window.create(sf::VideoMode(CollumsX * BlockSize.x, CollumsY * BlockSize.y), "Mesh Network");
        while (window.isOpen())
        {   sf::Event event;
            while (window.pollEvent(event))
            {
                //if (evnt.type == sf::Event::Closed) window.close();
                //zoom using sf::View::zoom and sf::Mouse::getPosition
                if (event.type == sf::Event::MouseWheelScrolled) {
                    if (event.mouseWheelScroll.wheel == sf::Mouse::VerticalWheel) {
                        if (event.mouseWheelScroll.delta > 0) {
                            sf::View view = window.getView();
                            view.zoom(0.9f);
                            window.setView(view);
                        } else {
                            sf::View view = window.getView();
                            view.zoom(1.1f);
                            window.setView(view);
                        }
                    }
                }
                //move using keyboard "w,a,s,d" and sf::View::move
                if (event.type == sf::Event::KeyPressed) {
                    if (event.key.code == sf::Keyboard::W) {
                        sf::View view = window.getView();
                        view.move(0.f, -10.f);
                        window.setView(view);
                    }
                    if (event.key.code == sf::Keyboard::A) {
                        sf::View view = window.getView();
                        view.move(-10.f, 0.f);
                        window.setView(view);
                    }
                    if (event.key.code == sf::Keyboard::S) {
                        sf::View view = window.getView();
                        view.move(0.f, 10.f);
                        window.setView(view);
                    }
                    if (event.key.code == sf::Keyboard::D) {
                        sf::View view = window.getView();
                        view.move(10.f, 0.f);
                        window.setView(view);
                    }
                }

                // delete nodes in range
                if (event.type == sf::Event::MouseButtonPressed) {
                    if (event.mouseButton.button == sf::Mouse::Left) {
                        sf::Vector2i mousePos = sf::Mouse::getPosition(window);
                        sf::Vector2f worldPos = window.mapPixelToCoords(mousePos);
                        int x1 = (int) (worldPos.x / 50.f);
                        int y1 = (int) (worldPos.y / 50.f);
                        int x2 = x1 + 2;
                        int y2 = y1 + 2;
                        deleteNodesInRange(x1, y1, x2, y2);
                    }
                }

                window.clear(sf::Color::Black);
                drawMesh();
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::R)) { Astar(grid[0][0], grid[n-1][n-1]); drawPath();}
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::B)) { BestFisrtSearch(grid[0][0], grid[n-1][n-1]); drawPath();}
                window.display();
            }
            //window.clear(sf::Color::Black);
            //drawMesh();
            //if (sf::Keyboard::isKeyPressed(sf::Keyboard::A)) { Astar(grid[0][0], grid[n-1][n-1]); drawPath();}
            ///if (sf::Keyboard::isKeyPressed(sf::Keyboard::B)) { BestFisrtSearch(grid[0][0], grid[n-1][n-1]); drawPath();}
            //window.display();
        }
    }

    //to obtain the weight of the node at position (1,1)
    float getWeight(int x, int y){
        return grid[x][y]->weight;
    }

    //to obtain the heuristic of the node at position (0,1)
    float getHeuristic(int x, int y){
        return grid[x][y]->heuristic;
    }

};



int main(){
    MeshNetwork mesh(SIZE);
    std::cout << "The weight of the node at position (1,1) is: " << mesh.getWeight(1,1) << std::endl;
    std::cout << "The heuristic of the node at position (1, 1) is: " << mesh.getHeuristic(1,1) << std::endl;
    mesh.visualise();
}


