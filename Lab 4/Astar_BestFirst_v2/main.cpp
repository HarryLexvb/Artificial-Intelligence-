#include<iostream>
#include<SFML/Graphics.hpp>
#include <cmath>

constexpr float WIDTH = 700;
constexpr float HEIGHT = 700;

sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "A* Pathfinder", sf::Style::Close);
sf::Event evnt;

const sf::Vector2f BlockSize = { 8, 8 };
const size_t CollumsX = WIDTH / BlockSize.x;
const size_t CollumsY = HEIGHT / BlockSize.y;

const sf::Color OutlineBlockColor = sf::Color::Black;
const sf::Color WallColor = sf::Color::Black;
const sf::Color StartColor = sf::Color::Blue;
const sf::Color EndColor = sf::Color::Red;
const sf::Color PathColor = sf::Color::Green;

struct Node
{
    bool isWall = false;
    bool isVisited = false;
    float FCost;
    float HCost;
    int x;
    int y;
    float weight = 1.0f;
    std::vector<Node*> Neighbours;
    Node* parent;
};

std::vector<Node*> Nodes;
Node* Start;
Node* End;

void initNeighBours(Node* node)
{
    if (node->x > 0) node->Neighbours.push_back(&Nodes[node->y][node->x - 1]);
    if (node->y > 0) node->Neighbours.push_back(&Nodes[node->y - 1][node->x]);
    if (node->x < CollumsX - 1) node->Neighbours.push_back(&Nodes[node->y][node->x + 1]);
    if (node->y < CollumsY - 1) node->Neighbours.push_back(&Nodes[node->y + 1][node->x]);

    if (node->y > 0 && node->x > 0) node->Neighbours.push_back(&Nodes[node->y - 1][node->x - 1]);
    if (node->y < CollumsY - 1 && node->x > 0) node->Neighbours.push_back(&Nodes[node->y + 1][node->x - 1]);
    if (node->y > 0 && node->x < CollumsX - 1) node->Neighbours.push_back(&Nodes[node->y - 1][node->x + 1]);
    if (node->y < CollumsY - 1 && node->x < CollumsX - 1) node->Neighbours.push_back(&Nodes[node->y + 1][node->x + 1]);


}

void initDefNode(Node* node, const int& x, const int& y)
{
    node->x = x;
    node->y = y;
    node->isWall = false;
    node->isVisited = false;
    node->parent = nullptr;
}

void init()
{
    for (int y = 0; y < CollumsY; y++)
    {
        Nodes.push_back(new Node[CollumsX]);
        for (int x = 0; x < CollumsX; x++)
        {
            initDefNode(&Nodes[y][x], x, y);
        }
    }

    for (int y = 0; y < CollumsY; y++)
        for (int x = 0; x < CollumsX; x++)
        {
            initNeighBours(&Nodes[y][x]);
        }

    Start = &Nodes[0][0];
    End = &Nodes[CollumsY - 1][CollumsX - 1];
}

void AStarAlg()
{
    for (int y = 0; y < CollumsY; y++)
        for (int x = 0; x < CollumsX; x++)
        {
            Nodes[y][x].isVisited = false;
            Nodes[y][x].FCost = INFINITY;
            Nodes[y][x].HCost = INFINITY;
            Nodes[y][x].parent = nullptr;
        }

    auto GetDist = [](Node* P1, Node* P2) { return sqrt(pow(P2->x - P1->x, 2) + pow(P2->y - P1->y, 2)); };

    Node* CurrentNode = Start;
    Start->HCost = 0.0f;
    Start->FCost = GetDist(Start, End);

    std::vector<Node*> NodesToTest;
    NodesToTest.push_back(Start);

    while (!NodesToTest.empty() && CurrentNode != End)
    {
        std::sort(NodesToTest.begin(), NodesToTest.end(), [](const Node* a, const Node* b) { return a->FCost < b->FCost; });

        while (!NodesToTest.empty() && NodesToTest.front()->isVisited)
            NodesToTest.erase(NodesToTest.begin());

        if (NodesToTest.empty())
            break;

        CurrentNode = NodesToTest.front();
        CurrentNode->isVisited = true;


        for (auto nodeNeighbour : CurrentNode->Neighbours)
        {
            if (!nodeNeighbour->isVisited && !nodeNeighbour->isWall)
                NodesToTest.push_back(nodeNeighbour);

            auto BestNode = CurrentNode->HCost + GetDist(CurrentNode, nodeNeighbour);

            if (BestNode < nodeNeighbour->HCost)
            {
                nodeNeighbour->parent = CurrentNode;
                nodeNeighbour->HCost = BestNode;

                nodeNeighbour->FCost = nodeNeighbour->HCost + GetDist(nodeNeighbour, End);
            }
        }
    }
}

void BestFisrtSearch(){
    // best first searh algorithm is different from A* algorithm in that it does not use the FCost to determine the next node to visit
    // instead it uses the HCost to determine the next node to visit
    // this is a very simple algorithm and is not very efficient
    // it is used to show the difference between A* and best first search

    for (int y = 0; y < CollumsY; y++)
        for (int x = 0; x < CollumsX; x++)
        {
            Nodes[y][x].isVisited = false;
            Nodes[y][x].FCost = INFINITY;
            Nodes[y][x].HCost = INFINITY;
            Nodes[y][x].parent = nullptr;
        }

    auto GetDist = [](Node* P1, Node* P2) { return sqrt(pow(P2->x - P1->x, 2) + pow(P2->y - P1->y, 2)); };

    Node* CurrentNode = Start;
    Start->HCost = 0.0f;
    Start->FCost = GetDist(Start, End);

    std::vector<Node*> NodesToTest;
    NodesToTest.push_back(Start);

    while (!NodesToTest.empty() && CurrentNode != End)
    {
        std::sort(NodesToTest.begin(), NodesToTest.end(), [](const Node* a, const Node* b) { return a->HCost < b->HCost; });

        while (!NodesToTest.empty() && NodesToTest.front()->isVisited)
            NodesToTest.erase(NodesToTest.begin());

        if (NodesToTest.empty())
            break;

        CurrentNode = NodesToTest.front();
        CurrentNode->isVisited = true;


        for (auto nodeNeighbour : CurrentNode->Neighbours)
        {
            if (!nodeNeighbour->isVisited && !nodeNeighbour->isWall)
                NodesToTest.push_back(nodeNeighbour);

            auto BestNode = CurrentNode->HCost + GetDist(CurrentNode, nodeNeighbour);

            if (BestNode < nodeNeighbour->HCost)
            {
                nodeNeighbour->parent = CurrentNode;
                nodeNeighbour->HCost = BestNode;

                nodeNeighbour->FCost = nodeNeighbour->HCost + GetDist(nodeNeighbour, End);
            }
        }
    }
}

void update()
{
    static bool NeedsToBeUpdated = true;

    int x = sf::Mouse::getPosition(window).x / BlockSize.x;
    int y = sf::Mouse::getPosition(window).y / BlockSize.y;

    if (x >= 0 && x < CollumsX && y >= 0 && y < CollumsY)
    {
        if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left))
        {
            if (!Nodes[y][x].isWall && (x != End->x || y != End->y)) { Start = &Nodes[y][x]; NeedsToBeUpdated = true; }
        }
        if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Right))
        {
            if (!Nodes[y][x].isWall && ( x != Start->x || y != Start->y)) { End = &Nodes[y][x]; NeedsToBeUpdated = true; }
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
        {
            static int xPrev = x;
            static int yPrev = y;

            /*if((x != xPrev || y != yPrev) && (x != End->x || y != End->y) && (x != Start->x || y != Start->y))
                Nodes[y][x].isWall = !Nodes[y][x].isWall;

            xPrev = x;
            yPrev = y;

            NeedsToBeUpdated = true;*/
            //delete a RANGE of walls instead of just one wall
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    if (x + i < CollumsX && y + j < CollumsY)
                    {
                        if ((x + i != End->x || y + j != End->y) && (x + i != Start->x || y + j != Start->y))
                            Nodes[y + j][x + i].isWall = !Nodes[y + j][x + i].isWall;
                    }
                }
            }
            NeedsToBeUpdated = true;
        }
    }

    //if (NeedsToBeUpdated) { AStarAlg(); NeedsToBeUpdated = false; }
    //if(NeedsToBeUpdated) { BestFisrtSearch(); NeedsToBeUpdated = false; }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::A)) { AStarAlg(); }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::B)) { BestFisrtSearch(); }
}

void draw()
{
    for (size_t y = 0; y < CollumsY; y++)
    {
        for (size_t x = 0; x < CollumsX; x++)
        {
            sf::RectangleShape s;
            s.setSize(BlockSize);
            s.setPosition(x * BlockSize.x, y * BlockSize.y);

            if (Nodes[y][x].isWall) s.setFillColor(WallColor);

            if (y == End->y && x == End->x) s.setFillColor(EndColor);

            if (y == Start->y && x == Start->x) s.setFillColor(StartColor);

            s.setOutlineThickness(BlockSize.x / 15);
            s.setOutlineColor(OutlineBlockColor);

            window.draw(s);
        }
    }

    if (End != nullptr)
    {
        Node* p = End->parent;
        while (p != nullptr && p != Start)
        {
            sf::RectangleShape s;
            s.setSize({ BlockSize.x / 2, BlockSize.y / 2 });
            s.setFillColor(PathColor);
            s.setPosition(p->x * BlockSize.x + BlockSize.x / 4, p->y * BlockSize.y + BlockSize.y / 4);

            p = p->parent;

            window.draw(s);
        }
    }

}

int main()
{
    window.setFramerateLimit(120);

    init();

    while (window.isOpen())
    {
        while (window.pollEvent(evnt))
        {
            if (evnt.type == sf::Event::Closed) window.close();
        }

        update();

        window.clear(sf::Color::White);

        draw();

        window.display();
    }
}