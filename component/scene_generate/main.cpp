#include <SFML/Graphics.hpp>
#include <vector>
#include <random>
#include <cmath>
#include <memory>
#include <queue>
#include <set>
#include <map>
#include <algorithm>
#include <iostream>

const int SCREEN_WIDTH = 800;
const int SCREEN_HEIGHT = 600;
const int CELL_SIZE = 10;
const int MAP_WIDTH = SCREEN_WIDTH / CELL_SIZE;
const int MAP_HEIGHT = SCREEN_HEIGHT / CELL_SIZE;

// 随机数生成器
std::random_device rd;
std::mt19937 gen(rd());

class Room {
public:
    int x1, y1, x2, y2;

    Room(int x, int y, int w, int h)
        : x1(x), y1(y), x2(x + w), y2(y + h) {}

    std::pair<int, int> center() const {
        return {(x1 + x2) / 2, (y1 + y2) / 2};
    }

    bool intersect(const Room& other) const {
        return (x1 <= other.x2 && x2 >= other.x1 &&
                y1 <= other.y2 && y2 >= other.y1);
    }
};

class Leaf {
public:
    int x, y, width, height;
    static const int MIN_LEAF_SIZE = 10;
    std::unique_ptr<Leaf> child1;
    std::unique_ptr<Leaf> child2;
    std::unique_ptr<Room> room;

    Leaf(int x, int y, int width, int height)
        : x(x), y(y), width(width), height(height) {}

    bool split() {
        if (child1 || child2)
            return false;

        bool splitH = std::uniform_real_distribution<>(0, 1)(gen) > 0.5;
        if (width > height && width / height >= 1.25)
            splitH = false;
        else if (height > width && height / width >= 1.25)
            splitH = true;

        int max = (splitH ? height : width) - MIN_LEAF_SIZE;
        if (max <= MIN_LEAF_SIZE)
            return false;

        int split = std::uniform_int_distribution<>(MIN_LEAF_SIZE, max)(gen);

        if (splitH) {
            child1 = std::make_unique<Leaf>(x, y, width, split);
            child2 = std::make_unique<Leaf>(x, y + split, width, height - split);
        } else {
            child1 = std::make_unique<Leaf>(x, y, split, height);
            child2 = std::make_unique<Leaf>(x + split, y, width - split, height);
        }

        return true;
    }

    ~Leaf() {
        printf("Released!motherfucker!!!\n");
    }
};

class DungeonGenerator {
protected:
    std::vector<std::vector<int>> map;

public:
    DungeonGenerator() : map(MAP_WIDTH, std::vector<int>(MAP_HEIGHT, 1)) {}

    virtual void generate() = 0;
    virtual DungeonGenerator* clone() const = 0;
    const std::vector<std::vector<int>>& getMap() const { return map; }

protected:
    void clearMap() {
        for (auto& row : map) {
            std::fill(row.begin(), row.end(), 1);
        }
    }
};

class TunnelingGenerator : public DungeonGenerator {
private:
    const int ROOM_MAX_SIZE = 15;
    const int ROOM_MIN_SIZE = 6;
    const int MAX_ROOMS = 30;
    std::vector<Room> rooms;

public:
    DungeonGenerator* clone() const override {
        return new TunnelingGenerator(*this);
    }
    void generate() override {
        clearMap();
        rooms.clear();

        for (int i = 0; i < MAX_ROOMS; i++) {
            int w = std::uniform_int_distribution<>(ROOM_MIN_SIZE, ROOM_MAX_SIZE)(gen);
            int h = std::uniform_int_distribution<>(ROOM_MIN_SIZE, ROOM_MAX_SIZE)(gen);
            int x = std::uniform_int_distribution<>(1, MAP_WIDTH - w - 1)(gen);
            int y = std::uniform_int_distribution<>(1, MAP_HEIGHT - h - 1)(gen);

            Room newRoom(x, y, w, h);

            bool failed = false;
            for (const auto& otherRoom : rooms) {
                if (newRoom.intersect(otherRoom)) {
                    failed = true;
                    break;
                }
            }

            if (!failed) {
                createRoom(newRoom);
                if (!rooms.empty()) {
                    auto [prevCenterX, prevCenterY] = rooms.back().center();
                    auto [newCenterX, newCenterY] = newRoom.center();

                    if (std::uniform_real_distribution<>(0, 1)(gen) > 0.5) {
                        createHorizontalTunnel(prevCenterX, newCenterX, prevCenterY);
                        createVerticalTunnel(prevCenterY, newCenterY, newCenterX);
                    } else {
                        createVerticalTunnel(prevCenterY, newCenterY, prevCenterX);
                        createHorizontalTunnel(prevCenterX, newCenterX, newCenterY);
                    }
                }
                rooms.push_back(newRoom);
            }
        }
    }

private:
    void createRoom(const Room& room) {
        for (int x = room.x1 + 1; x < room.x2; x++) {
            for (int y = room.y1 + 1; y < room.y2; y++) {
                if (x > 0 && x < MAP_WIDTH && y > 0 && y < MAP_HEIGHT) {
                    map[x][y] = 0;
                }
            }
        }
    }

    void createHorizontalTunnel(int x1, int x2, int y) {
        for (int x = std::min(x1, x2); x <= std::max(x1, x2); x++) {
            if (x > 0 && x < MAP_WIDTH && y > 0 && y < MAP_HEIGHT) {
                map[x][y] = 0;
            }
        }
    }

    void createVerticalTunnel(int y1, int y2, int x) {
        for (int y = std::min(y1, y2); y <= std::max(y1, y2); y++) {
            if (x > 0 && x < MAP_WIDTH && y > 0 && y < MAP_HEIGHT) {
                map[x][y] = 0;
            }
        }
    }
};

class BSPGenerator : public DungeonGenerator {
private:
    const int MAX_LEAF_SIZE = 24;
    const int ROOM_MAX_SIZE = 15;
    const int ROOM_MIN_SIZE = 6;
    std::vector<Leaf*> leafs;

public:
    DungeonGenerator* clone() const override {
        return new BSPGenerator(*this);
    }

    void generate() override {
        leafs.reserve(512);
        clearMap();
        leafs.clear();

        auto rootLeaf = std::make_unique<Leaf>(0, 0, MAP_WIDTH, MAP_HEIGHT);
        leafs.push_back(rootLeaf.get());

        bool splitSuccessfully = true;
        while (splitSuccessfully) {
            splitSuccessfully = false;
            for (auto leaf : leafs) {
                if (!leaf->child1 && !leaf->child2) {
                    if (leaf->width > MAX_LEAF_SIZE ||
                        leaf->height > MAX_LEAF_SIZE ||
                        std::uniform_real_distribution<>(0, 1)(gen) > 0.8) {
                        if (leaf->split()) {
                            leafs.push_back(leaf->child1.get());
                            leafs.push_back(leaf->child2.get());
                            splitSuccessfully = true;
                        }
                    }
                }
            }
        }

        createRooms();
    }

private:
    void createRooms() {
        for (auto& leaf : leafs) {
            if (!leaf->child1 && !leaf->child2) {
                int w = std::uniform_int_distribution<>(ROOM_MIN_SIZE,
                    std::min(ROOM_MAX_SIZE, leaf->width - 1))(gen);
                int h = std::uniform_int_distribution<>(ROOM_MIN_SIZE,
                    std::min(ROOM_MAX_SIZE, leaf->height - 1))(gen);
                int x = std::uniform_int_distribution<>(
                    leaf->x, leaf->x + (leaf->width - 1) - w)(gen);
                int y = std::uniform_int_distribution<>(
                    leaf->y, leaf->y + (leaf->height - 1) - h)(gen);

                leaf->room = std::make_unique<Room>(x, y, w, h);
                createRoom(*leaf->room);
            }
        }

        // Connect rooms
        for (size_t i = 0; i < leafs.size() - 1; i++) {
            if (leafs[i]->room && leafs[i + 1]->room) {
                connectRooms(*leafs[i]->room, *leafs[i + 1]->room);
            }
        }
    }

    void createRoom(const Room& room) {
        for (int x = room.x1 + 1; x < room.x2; x++) {
            for (int y = room.y1 + 1; y < room.y2; y++) {
                if (x > 0 && x < MAP_WIDTH && y > 0 && y < MAP_HEIGHT) {
                    map[x][y] = 0;
                }
            }
        }
    }

    void connectRooms(const Room& room1, const Room& room2) {
        auto [x1, y1] = room1.center();
        auto [x2, y2] = room2.center();

        if (std::uniform_real_distribution<>(0, 1)(gen) > 0.5) {
            createHorizontalTunnel(x1, x2, y1);
            createVerticalTunnel(y1, y2, x2);
        } else {
            createVerticalTunnel(y1, y2, x1);
            createHorizontalTunnel(x1, x2, y2);
        }
    }

    void createHorizontalTunnel(int x1, int x2, int y) {
        for (int x = std::min(x1, x2); x <= std::max(x1, x2); x++) {
            if (x > 0 && x < MAP_WIDTH && y > 0 && y < MAP_HEIGHT) {
                map[x][y] = 0;
            }
        }
    }

    void createVerticalTunnel(int y1, int y2, int x) {
        for (int y = std::min(y1, y2); y <= std::max(y1, y2); y++) {
            if (x > 0 && x < MAP_WIDTH && y > 0 && y < MAP_HEIGHT) {
                map[x][y] = 0;
            }
        }
    }
};

class DrunkardWalkGenerator : public DungeonGenerator {
private:
    const float PERCENT_GOAL = 0.4f;
    const int WALK_ITERATIONS = 25000;
    const float WEIGHTED_TOWARD_CENTER = 0.15f;
    const float WEIGHTED_TOWARD_PREVIOUS_DIRECTION = 0.7f;
    int filled;
    std::string previousDirection;
    int drunkardX, drunkardY;

public:
    DungeonGenerator* clone() const override {
        return new DrunkardWalkGenerator(*this);
    }

    void generate() override {
        clearMap();
        filled = 0;
        previousDirection = "";

        drunkardX = std::uniform_int_distribution<>(2, MAP_WIDTH - 3)(gen);
        drunkardY = std::uniform_int_distribution<>(2, MAP_HEIGHT - 3)(gen);

        int filledGoal = MAP_WIDTH * MAP_HEIGHT * PERCENT_GOAL;

        for (int i = 0; i < WALK_ITERATIONS && filled < filledGoal; i++) {
            walk();
        }
    }

private:
    void walk() {
        float north = 1.0f;
        float south = 1.0f;
        float east = 1.0f;
        float west = 1.0f;

        // Weight against edges
        if (drunkardX < MAP_WIDTH * 0.25) {
            east += WEIGHTED_TOWARD_CENTER;
        } else if (drunkardX > MAP_WIDTH * 0.75) {
            west += WEIGHTED_TOWARD_CENTER;
        }
        if (drunkardY < MAP_HEIGHT * 0.25) {
            south += WEIGHTED_TOWARD_CENTER;
        } else if (drunkardY > MAP_HEIGHT * 0.75) {
            north += WEIGHTED_TOWARD_CENTER;
        }

        // Weight toward previous direction
        if (previousDirection == "north") north += WEIGHTED_TOWARD_PREVIOUS_DIRECTION;
        if (previousDirection == "south") south += WEIGHTED_TOWARD_PREVIOUS_DIRECTION;
        if (previousDirection == "east") east += WEIGHTED_TOWARD_PREVIOUS_DIRECTION;
        if (previousDirection == "west") west += WEIGHTED_TOWARD_PREVIOUS_DIRECTION;

        // Normalize probabilities
        float total = north + south + east + west;
        north /= total;
        south /= total;
        east /= total;
        west /= total;

        // Choose direction
        float choice = std::uniform_real_distribution<>(0, 1)(gen);
        int dx = 0, dy = 0;

        if (choice < north) {
            dy = -1;
            previousDirection = "north";
        } else if (choice < north + south) {
            dy = 1;
            previousDirection = "south";
        } else if (choice < north + south + east) {
            dx = 1;
            previousDirection = "east";
        } else {
            dx = -1;
            previousDirection = "west";
        }

        // Move drunkard
        if (drunkardX + dx > 0 && drunkardX + dx < MAP_WIDTH - 1 &&
            drunkardY + dy > 0 && drunkardY + dy < MAP_HEIGHT - 1) {
            drunkardX += dx;
            drunkardY += dy;
            if (map[drunkardX][drunkardY] == 1) {
                map[drunkardX][drunkardY] = 0;
                filled++;
            }
        }
    }
};

class CellularAutomataGenerator : public DungeonGenerator {
private:
    const int ITERATIONS = 50000;
    const int NEIGHBORS = 4;
    const float WALL_PROBABILITY = 0.5f;
    const int ROOM_MIN_SIZE = 16;
    const int ROOM_MAX_SIZE = 500;
    const bool SMOOTH_EDGES = true;
    const int SMOOTHING = 1;
    std::vector<std::set<std::pair<int,int>>> caves;

public:
    CellularAutomataGenerator* clone() const override {
        return new CellularAutomataGenerator(*this);
    }

    void generate() override {
        clearMap();
        caves.clear();

        // Random fill
        randomFillMap();

        // Create caves
        createCaves();

        // Get and connect caves
        getCaves();
        connectCaves();

        // Clean up
        cleanUpMap();
    }

private:
    void randomFillMap() {
        for(int y = 1; y < MAP_HEIGHT-1; y++) {
            for(int x = 1; x < MAP_WIDTH-1; x++) {
                if(std::uniform_real_distribution<>(0,1)(gen) >= WALL_PROBABILITY) {
                    map[x][y] = 0;
                }
            }
        }
    }

    void createCaves() {
        for(int i = 0; i < ITERATIONS; i++) {
            int tileX = std::uniform_int_distribution<>(1, MAP_WIDTH-2)(gen);
            int tileY = std::uniform_int_distribution<>(1, MAP_HEIGHT-2)(gen);

            if(getAdjacentWalls(tileX, tileY) > NEIGHBORS) {
                map[tileX][tileY] = 1;
            } else if(getAdjacentWalls(tileX, tileY) < NEIGHBORS) {
                map[tileX][tileY] = 0;
            }
        }
        cleanUpMap();
    }

    void cleanUpMap() {
        if(SMOOTH_EDGES) {
            for(int i = 0; i < 5; i++) {
                for(int x = 1; x < MAP_WIDTH-1; x++) {
                    for(int y = 1; y < MAP_HEIGHT-1; y++) {
                        if(map[x][y] == 1 && getAdjacentWallsSimple(x,y) <= SMOOTHING) {
                            map[x][y] = 0;
                        }
                    }
                }
            }
        }
    }

    void createTunnel(const std::set<std::pair<int,int>>& cave1,
                    const std::pair<int,int>& point1,
                    const std::pair<int,int>& point2) {
        int x = point2.first;
        int y = point2.second;

        while(true) {
            if(cave1.find(std::make_pair(x,y)) != cave1.end()) {
                break;
            }

            // Calculate weights
            double north = 1.0, south = 1.0, east = 1.0, west = 1.0;
            const double weight = 1.0;

            if(x < point1.first) east += weight;
            else if(x > point1.first) west += weight;
            if(y < point1.second) south += weight;
            else if(y > point1.second) north += weight;

            // Normalize weights
            double total = north + south + east + west;
            north /= total;
            south /= total;
            east /= total;
            west /= total;

            // Choose direction
            double choice = std::uniform_real_distribution<>(0,1)(gen);
            int dx = 0, dy = 0;

            if(choice < north) {
                dy = -1;
            } else if(choice < north + south) {
                dy = 1;
            } else if(choice < north + south + east) {
                dx = 1;
            } else {
                dx = -1;
            }

            // Check bounds and move
            if(x + dx > 0 && x + dx < MAP_WIDTH-1 && y + dy > 0 && y + dy < MAP_HEIGHT-1) {
                x += dx;
                y += dy;
                map[x][y] = 0;
            }
        }
    }


    int getAdjacentWallsSimple(int x, int y) {
        int wallCount = 0;
        if(map[x][y-1] == 1) wallCount++;
        if(map[x][y+1] == 1) wallCount++;
        if(map[x-1][y] == 1) wallCount++;
        if(map[x+1][y] == 1) wallCount++;
        return wallCount;
    }

    int getAdjacentWalls(int x, int y) {
        int wallCount = 0;
        for(int i = -1; i <= 1; i++) {
            for(int j = -1; j <= 1; j++) {
                if(i == 0 && j == 0) continue;
                int checkX = x + i;
                int checkY = y + j;
                if(checkX > 0 && checkX < MAP_WIDTH && checkY > 0 && checkY < MAP_HEIGHT) {
                    wallCount += map[checkX][checkY];
                }
            }
        }
        return wallCount;
    }


    void getCaves() {
        for(int x = 0; x < MAP_WIDTH; x++) {
            for(int y = 0; y < MAP_HEIGHT; y++) {
                if(map[x][y] == 0) {
                    floodFill(x, y);
                }
            }
        }

        for(const auto& cave : caves) {
            for(const auto& tile : cave) {
                map[tile.first][tile.second] = 0;
            }
        }
    }

    void floodFill(int startX, int startY) {
        std::set<std::pair<int,int>> cave;
        std::set<std::pair<int,int>> toBeFilled;
        toBeFilled.insert({startX, startY});

        while(!toBeFilled.empty()) {
            auto tile = *toBeFilled.begin();
            toBeFilled.erase(toBeFilled.begin());

            if(cave.find(tile) == cave.end()) {
                cave.insert(tile);
                map[tile.first][tile.second] = 1;

                // Check adjacent cells
                const std::vector<std::pair<int,int>> directions = {
                    {0,-1}, {0,1}, {1,0}, {-1,0}
                };

                for(const auto& dir : directions) {
                    int newX = tile.first + dir.first;
                    int newY = tile.second + dir.second;
                    if(newX > 0 && newX < MAP_WIDTH && newY > 0 && newY < MAP_HEIGHT) {
                        if(map[newX][newY] == 0) {
                            std::pair<int,int> newTile = {newX, newY};
                            if(toBeFilled.find(newTile) == toBeFilled.end() &&
                               cave.find(newTile) == cave.end()) {
                                toBeFilled.insert(newTile);
                            }
                        }
                    }
                }
            }
        }

        if(cave.size() >= ROOM_MIN_SIZE) {
            caves.push_back(cave);
        }
    }

    void connectCaves() {
        for(auto& currentCave : caves) {
            std::pair<int,int> point1 = *currentCave.begin();
            std::pair<int,int> point2;
            const std::set<std::pair<int,int>>* targetCave = nullptr;
            double minDistance = std::numeric_limits<double>::max();

            for(auto& nextCave : caves) {
                if(&nextCave != &currentCave && !checkConnectivity(currentCave, nextCave)) {
                    std::pair<int,int> nextPoint = *nextCave.begin();
                    double dist = distanceFormula(point1, nextPoint);
                    if(dist < minDistance) {
                        point2 = nextPoint;
                        minDistance = dist;
                        targetCave = &nextCave;
                    }
                }
            }

            if(targetCave) {
                createTunnel(currentCave, point1, point2);
            }
        }
    }


    bool checkConnectivity(const std::set<std::pair<int,int>>& cave1,
                        const std::set<std::pair<int,int>>& cave2) {
        std::set<std::pair<int,int>> connected;
        std::set<std::pair<int,int>> toCheck;
        toCheck.insert(*cave1.begin());

        while(!toCheck.empty()) {
            auto current = *toCheck.begin();
            toCheck.erase(toCheck.begin());

            if(connected.find(current) == connected.end()) {
                connected.insert(current);

                // Check adjacent cells
                const std::vector<std::pair<int,int>> dirs = {
                    {0,-1}, {0,1}, {1,0}, {-1,0}
                };

                for(const auto& dir : dirs) {
                    int newX = current.first + dir.first;
                    int newY = current.second + dir.second;
                    std::pair<int,int> next{newX, newY};

                    if(map[newX][newY] == 0 &&
                    toCheck.find(next) == toCheck.end() &&
                    connected.find(next) == connected.end()) {
                        toCheck.insert(next);
                    }
                }
            }
        }

        return connected.find(*cave2.begin()) != connected.end();
    }

    double distanceFormula(const std::pair<int,int>& p1,
                        const std::pair<int,int>& p2) {
        return std::sqrt(std::pow(p2.first - p1.first, 2) +
                        std::pow(p2.second - p1.second, 2));
    }

};

class RoomAdditionGenerator : public DungeonGenerator {
private:
    const int ROOM_MAX_SIZE = 18;
    const int ROOM_MIN_SIZE = 16;
    const int MAX_NUM_ROOMS = 30;
    const int SQUARE_ROOM_MAX_SIZE = 12;
    const int SQUARE_ROOM_MIN_SIZE = 6;
    const int CROSS_ROOM_MAX_SIZE = 12;
    const int CROSS_ROOM_MIN_SIZE = 6;
    const float CAVERN_CHANCE = 0.4f;
    const int CAVERN_MAX_SIZE = 35;
    const float WALL_PROBABILITY = 0.45f;
    const int NEIGHBORS = 4;
    const float SQUARE_ROOM_CHANCE = 0.2f;
    const float CROSS_ROOM_CHANCE = 0.15f;
    const int BUILD_ROOM_ATTEMPTS = 500;
    const int PLACE_ROOM_ATTEMPTS = 20;
    const int MAX_TUNNEL_LENGTH = 12;

    std::vector<std::vector<int>> currentRoom;
    std::vector<std::vector<std::vector<int>>> rooms;

public:
    DungeonGenerator* clone() const override {
        return new RoomAdditionGenerator(*this);
    }

    void generate() override {
        clearMap();
        rooms.clear();

        // Generate first room
        auto firstRoom = generateRoom();
        auto [roomWidth, roomHeight] = getRoomDimensions(firstRoom);
        int roomX = (MAP_WIDTH/2 - roomWidth/2) - 1;
        int roomY = (MAP_HEIGHT/2 - roomHeight/2) - 1;
        addRoom(roomX, roomY, firstRoom);

        // Generate other rooms
        for(int i = 0; i < BUILD_ROOM_ATTEMPTS; i++) {
            auto room = generateRoom();
            auto [x, y, wallTile, direction, length] = placeRoom(room);

            if(x != -1 && y != -1) {
                addRoom(x, y, room);
                addTunnel(wallTile, direction, length);

                if(rooms.size() >= MAX_NUM_ROOMS) {
                    break;
                }
            }
        }
    }

private:
    std::vector<std::vector<int>> generateRoom() {
        if(!rooms.empty()) {
            float choice = std::uniform_real_distribution<>(0,1)(gen);

            if(choice < SQUARE_ROOM_CHANCE) {
                return generateSquareRoom();
            } else if(choice < SQUARE_ROOM_CHANCE + CROSS_ROOM_CHANCE) {
                return generateCrossRoom();
            } else {
                return generateCellularRoom();
            }
        } else {
            if(std::uniform_real_distribution<>(0,1)(gen) < CAVERN_CHANCE) {
                return generateCavernRoom();
            } else {
                return generateSquareRoom();
            }
        }
    }

    std::vector<std::vector<int>> generateSquareRoom() {
        int width = std::uniform_int_distribution<>(
            SQUARE_ROOM_MIN_SIZE, SQUARE_ROOM_MAX_SIZE)(gen);
        int height = std::uniform_int_distribution<>(
            std::max(int(width*0.5), SQUARE_ROOM_MIN_SIZE),
            std::min(int(width*1.5), SQUARE_ROOM_MAX_SIZE))(gen);

        std::vector<std::vector<int>> room(width, std::vector<int>(height, 1));

        // Create interior
        for(int x = 1; x < width-1; x++) {
            for(int y = 1; y < height-1; y++) {
                room[x][y] = 0;
            }
        }

        return room;
    }

    std::vector<std::vector<int>> generateCrossRoom() {
        int horWidth = ((std::uniform_int_distribution<>(
            CROSS_ROOM_MIN_SIZE+2, CROSS_ROOM_MAX_SIZE)(gen))/2)*2;
        int virHeight = ((std::uniform_int_distribution<>(
            CROSS_ROOM_MIN_SIZE+2, CROSS_ROOM_MAX_SIZE)(gen))/2)*2;
        int horHeight = ((std::uniform_int_distribution<>(
            CROSS_ROOM_MIN_SIZE, virHeight-2)(gen))/2)*2;
        int virWidth = ((std::uniform_int_distribution<>(
            CROSS_ROOM_MIN_SIZE, horWidth-2)(gen))/2)*2;

        std::vector<std::vector<int>> room(horWidth, std::vector<int>(virHeight, 1));

        // Fill horizontal space
        int virOffset = (virHeight/2 - horHeight/2);
        for(int y = virOffset; y < horHeight+virOffset; y++) {
            for(int x = 0; x < horWidth; x++) {
                room[x][y] = 0;
            }
        }

        // Fill vertical space
        int horOffset = (horWidth/2 - virWidth/2);
        for(int y = 0; y < virHeight; y++) {
            for(int x = horOffset; x < virWidth+horOffset; x++) {
                room[x][y] = 0;
            }
        }

        return room;
    }

    std::vector<std::vector<int>> generateCellularRoom() {
        while(true) {
            std::vector<std::vector<int>> room(
                ROOM_MAX_SIZE, std::vector<int>(ROOM_MAX_SIZE, 1));

            // Random fill
            for(int y = 2; y < ROOM_MAX_SIZE-2; y++) {
                for(int x = 2; x < ROOM_MAX_SIZE-2; x++) {
                    if(std::uniform_real_distribution<>(0,1)(gen) >= WALL_PROBABILITY) {
                        room[x][y] = 0;
                    }
                }
            }

            // Cellular automata iterations
            for(int i = 0; i < 4; i++) {
                for(int y = 1; y < ROOM_MAX_SIZE-1; y++) {
                    for(int x = 1; x < ROOM_MAX_SIZE-1; x++) {
                        int walls = countAdjacentWalls(x, y, room);
                        if(walls > NEIGHBORS) {
                            room[x][y] = 1;
                        } else if(walls < NEIGHBORS) {
                            room[x][y] = 0;
                        }
                    }
                }
            }

            // Check if room is valid
            auto [width, height] = getRoomDimensions(room);
            for(int x = 0; x < width; x++) {
                for(int y = 0; y < height; y++) {
                    if(room[x][y] == 0) {
                        return room;
                    }
                }
            }
        }
    }

    std::vector<std::vector<int>> generateCavernRoom() {
        // Similar to generateCellularRoom but with CAVERN_MAX_SIZE
        // Implementation similar to cellular room but larger
        return generateCellularRoom(); // Simplified for this example
    }

    void addRoom(int roomX, int roomY, const std::vector<std::vector<int>>& room) {
        auto [width, height] = getRoomDimensions(room);
        for(int x = 0; x < width; x++) {
            for(int y = 0; y < height; y++) {
                if(room[x][y] == 0) {
                    map[roomX+x][roomY+y] = 0;
                }
            }
        }
        rooms.push_back(room);
    }

    void addTunnel(std::pair<int,int> wallTile, std::pair<int,int> direction, int tunnelLength) {
        int startX = wallTile.first + direction.first * tunnelLength;
        int startY = wallTile.second + direction.second * tunnelLength;

        for(int i = 0; i < MAX_TUNNEL_LENGTH; i++) {
            int x = startX - direction.first * i;
            int y = startY - direction.second * i;
            map[x][y] = 0;

            if(x + direction.first == wallTile.first &&
               y + direction.second == wallTile.second) {
                break;
            }
        }
    }

    std::pair<int,int> getDirection() {
        const std::vector<std::pair<int,int>> directions = {
            {0,-1}, {0,1}, {1,0}, {-1,0}  // North, South, East, West
        };
        return directions[std::uniform_int_distribution<>(0,3)(gen)];
    }

    std::tuple<int,int,std::pair<int,int>,std::pair<int,int>,int> placeRoom(
        const std::vector<std::vector<int>>& room) {

        auto [roomWidth, roomHeight] = getRoomDimensions(room);

        for(int attempt = 0; attempt < PLACE_ROOM_ATTEMPTS; attempt++) {
            auto direction = getDirection();

            // Find valid wall tile
            std::pair<int,int> wallTile;
            bool found = false;

            for(int tries = 0; tries < 100 && !found; tries++) {
                int x = std::uniform_int_distribution<>(1, MAP_WIDTH-2)(gen);
                int y = std::uniform_int_distribution<>(1, MAP_HEIGHT-2)(gen);

                if(map[x][y] == 1 &&
                   map[x+direction.first][y+direction.second] == 1 &&
                   map[x-direction.first][y-direction.second] == 0) {
                    wallTile = {x,y};
                    found = true;
                }
            }

            if(!found) continue;

            // Try to place room
            for(int tunnelLength = 0; tunnelLength < MAX_TUNNEL_LENGTH; tunnelLength++) {
                int roomX = wallTile.first - direction.first * tunnelLength;
                int roomY = wallTile.second - direction.second * tunnelLength;

                if(canPlaceRoom(room, roomX, roomY)) {
                    return {roomX, roomY, wallTile, direction, tunnelLength};
                }
            }
        }

        return {-1, -1, {-1,-1}, {-1,-1}, -1};
    }

    bool canPlaceRoom(const std::vector<std::vector<int>>& room, int roomX, int roomY) {
        auto [width, height] = getRoomDimensions(room);

        // Check bounds
        if(roomX < 1 || roomX + width >= MAP_WIDTH-1 ||
           roomY < 1 || roomY + height >= MAP_HEIGHT-1) {
            return false;
        }

        // Check overlap
        for(int x = -1; x <= width; x++) {
            for(int y = -1; y <= height; y++) {
                if(map[roomX+x][roomY+y] == 0) {
                    return false;
                }
            }
        }

        return true;
    }

    std::pair<int,int> getRoomDimensions(const std::vector<std::vector<int>>& room) {
        if(room.empty()) return {0,0};
        return {static_cast<int>(room.size()), static_cast<int>(room[0].size())};
    }

    int countAdjacentWalls(int x, int y, const std::vector<std::vector<int>>& room) {
        int count = 0;
        for(int i = -1; i <= 1; i++) {
            for(int j = -1; j <= 1; j++) {
                if(i == 0 && j == 0) continue;
                count += room[x+i][y+j];
            }
        }
        return count;
    }
};

class CityWallsGenerator : public DungeonGenerator {
private:
    const int MAX_LEAF_SIZE = 30;
    const int ROOM_MAX_SIZE = 16;
    const int ROOM_MIN_SIZE = 8;
    std::vector<Leaf*> leafs;
    std::vector<Room> rooms;

public:
    DungeonGenerator* clone() const override {
        return new CityWallsGenerator(*this);
    }

    void generate() override {
        leafs.reserve(512);
        clearMap();
        leafs.clear();
        rooms.clear();

        // Start with all floors
        for(auto& row : map) {
            std::fill(row.begin(), row.end(), 0);
        }

        // Create BSP tree
        auto rootLeaf = std::make_unique<Leaf>(0, 0, MAP_WIDTH, MAP_HEIGHT);
        leafs.push_back(rootLeaf.get());

        bool splitSuccessfully = true;
        while(splitSuccessfully) {
            splitSuccessfully = false;
            for(auto leaf : leafs) {
                if(!leaf->child1 && !leaf->child2) {
                    if(leaf->width > MAX_LEAF_SIZE ||
                       leaf->height > MAX_LEAF_SIZE ||
                       std::uniform_real_distribution<>(0,1)(gen) > 0.8) {
                        if(leaf->split()) {
                            leafs.push_back(leaf->child1.get());
                            leafs.push_back(leaf->child2.get());
                            splitSuccessfully = true;
                        }
                    }
                }
            }
        }

        createRooms();
        createDoors();
    }

private:
    void createRooms() {
        for(auto& leaf : leafs) {
            if(!leaf->child1 && !leaf->child2) {
                int w = std::uniform_int_distribution<>(
                    ROOM_MIN_SIZE,
                    std::min(ROOM_MAX_SIZE, leaf->width-1))(gen);
                int h = std::uniform_int_distribution<>(
                    ROOM_MIN_SIZE,
                    std::min(ROOM_MAX_SIZE, leaf->height-1))(gen);
                int x = std::uniform_int_distribution<>(
                    leaf->x, leaf->x + (leaf->width-1) - w)(gen);
                int y = std::uniform_int_distribution<>(
                    leaf->y, leaf->y + (leaf->height-1) - h)(gen);

                createRoom(Room(x, y, w, h));
                rooms.push_back(Room(x, y, w, h));
            }
        }
    }

    void createRoom(const Room& room) {
        // Build Walls
        for(int x = room.x1 + 1; x < room.x2; x++) {
            for(int y = room.y1 + 1; y < room.y2; y++) {
                map[x][y] = 1;
            }
        }

        // Build Interior
        for(int x = room.x1 + 2; x < room.x2 - 1; x++) {
            for(int y = room.y1 + 2; y < room.y2 - 1; y++) {
                map[x][y] = 0;
            }
        }
    }

    void createDoors() {
        for(auto& room : rooms) {
            auto [centerX, centerY] = room.center();

            enum class Wall { North, South, East, West };
            Wall wall = static_cast<Wall>(
                std::uniform_int_distribution<>(0,3)(gen));

            int doorX, doorY;
            switch(wall) {
                case Wall::North:
                    doorX = centerX;
                    doorY = room.y1 + 1;
                    break;
                case Wall::South:
                    doorX = centerX;
                    doorY = room.y2 - 1;
                    break;
                case Wall::East:
                    doorX = room.x2 - 1;
                    doorY = centerY;
                    break;
                case Wall::West:
                    doorX = room.x1 + 1;
                    doorY = centerY;
                    break;
            }

            map[doorX][doorY] = 0;
        }
    }
};

class MazeWithRoomsGenerator : public DungeonGenerator {
private:
    const int ROOM_MAX_SIZE = 13;
    const int ROOM_MIN_SIZE = 6;
    const int BUILD_ROOM_ATTEMPTS = 100;
    const float CONNECTION_CHANCE = 0.04f;
    const float WINDING_PERCENT = 0.1f;
    const bool ALLOW_DEAD_ENDS = false;

    std::vector<std::vector<int>> regions;
    int currentRegion;

public:
    DungeonGenerator* clone() const override {
        return new MazeWithRoomsGenerator(*this);
    }

    void generate() override {
        clearMap();

        // Ensure odd dimensions
        int adjustedWidth = (MAP_WIDTH % 2 == 0) ? MAP_WIDTH - 1 : MAP_WIDTH;
        int adjustedHeight = (MAP_HEIGHT % 2 == 0) ? MAP_HEIGHT - 1 : MAP_HEIGHT;

        // Initialize regions
        regions = std::vector<std::vector<int>>(
            adjustedWidth, std::vector<int>(adjustedHeight, -1));
        currentRegion = -1;

        // Add rooms
        addRooms(adjustedWidth, adjustedHeight);

        // Generate maze in remaining space
        for(int y = 1; y < adjustedHeight; y += 2) {
            for(int x = 1; x < adjustedWidth; x += 2) {
                if(map[x][y] != 1) continue;
                growMaze({x, y}, adjustedWidth, adjustedHeight);
            }
        }

        // Connect regions
        connectRegions(adjustedWidth, adjustedHeight);

        // Remove dead ends if needed
        if(!ALLOW_DEAD_ENDS) {
            removeDeadEnds(adjustedWidth, adjustedHeight);
        }
    }

private:
    void addRooms(int mapWidth, int mapHeight) {
        std::vector<Room> rooms;

        for(int i = 0; i < BUILD_ROOM_ATTEMPTS; i++) {
            // Generate room with odd dimensions
            int width = (std::uniform_int_distribution<>(
                ROOM_MIN_SIZE/2, ROOM_MAX_SIZE/2)(gen) * 2) + 1;
            int height = (std::uniform_int_distribution<>(
                ROOM_MIN_SIZE/2, ROOM_MAX_SIZE/2)(gen) * 2) + 1;

            int x = ((std::uniform_int_distribution<>(
                0, mapWidth - width - 1)(gen)) / 2) * 2 + 1;
            int y = ((std::uniform_int_distribution<>(
                0, mapHeight - height - 1)(gen)) / 2) * 2 + 1;

            Room newRoom(x, y, width, height);

            bool failed = false;
            for(const auto& room : rooms) {
                if(newRoom.intersect(room)) {
                    failed = true;
                    break;
                }
            }

            if(!failed) {
                rooms.push_back(newRoom);
                startNewRegion();
                createRoom(newRoom);
            }
        }
    }

    void growMaze(std::pair<int,int> start, int mapWidth, int mapHeight) {
        std::vector<std::pair<int,int>> cells;
        std::pair<int,int> lastDir = {0,0};

        startNewRegion();
        carve(start.first, start.second);
        cells.push_back(start);

        while(!cells.empty()) {
            auto cell = cells.back();

            std::vector<std::pair<int,int>> unmadeCells;
            const std::vector<std::pair<int,int>> directions = {
                {0,-1}, {0,1}, {1,0}, {-1,0}
            };

            for(const auto& dir : directions) {
                if(canCarve(cell, dir, mapWidth, mapHeight)) {
                    unmadeCells.push_back(dir);
                }
            }

            if(!unmadeCells.empty()) {
                std::pair<int,int> dir;
                if(std::find(unmadeCells.begin(), unmadeCells.end(), lastDir) != unmadeCells.end()
                   && std::uniform_real_distribution<>(0,1)(gen) > WINDING_PERCENT) {
                    dir = lastDir;
                } else {
                    dir = unmadeCells[std::uniform_int_distribution<>(
                        0, unmadeCells.size()-1)(gen)];
                }

                auto newCell = std::make_pair(
                    cell.first + dir.first, cell.second + dir.second);
                carve(newCell.first, newCell.second);

                newCell = std::make_pair(
                    cell.first + dir.first*2, cell.second + dir.second*2);
                carve(newCell.first, newCell.second);
                cells.push_back(newCell);

                lastDir = dir;
            } else {
                cells.pop_back();
                lastDir = {0,0};
            }
        }
    }

    bool canCarve(std::pair<int,int> pos, std::pair<int,int> dir,
                  int mapWidth, int mapHeight) {
        int x = pos.first + dir.first * 3;
        int y = pos.second + dir.second * 3;

        if(x <= 0 || x >= mapWidth-1 || y <= 0 || y >= mapHeight-1) {
            return false;
        }

        x = pos.first + dir.first * 2;
        y = pos.second + dir.second * 2;

        return map[x][y] == 1;
    }

    void connectRegions(int mapWidth, int mapHeight) {
        std::vector<std::vector<std::set<int>>> connectorRegions(
            mapWidth, std::vector<std::set<int>>(mapHeight));

        // Find all connectors
        for(int x = 1; x < mapWidth-1; x++) {
            for(int y = 1; y < mapHeight-1; y++) {
                if(map[x][y] != 1) continue;

                std::set<int> regions;
                const std::vector<std::pair<int,int>> directions = {
                    {0,-1}, {0,1}, {1,0}, {-1,0}
                };

                for(const auto& dir : directions) {
                    int newX = x + dir.first;
                    int newY = y + dir.second;
                    int region = this->regions[newX][newY];
                    if(region != -1) regions.insert(region);
                }

                if(regions.size() < 2) continue;

                connectorRegions[x][y] = regions;
            }
        }

        // Connect regions
        std::map<int,int> merged;
        std::set<int> openRegions;
        for(int i = 0; i <= currentRegion; i++) {
            merged[i] = i;
            openRegions.insert(i);
        }

        while(openRegions.size() > 1) {
            // Find a connector
            int connX = -1, connY = -1;
            for(int x = 0; x < mapWidth; x++) {
                for(int y = 0; y < mapHeight; y++) {
                    if(!connectorRegions[x][y].empty()) {
                        connX = x;
                        connY = y;
                        break;
                    }
                }
                if(connX != -1) break;
            }

            if(connX == -1) break;

            // Merge regions
            map[connX][connY] = 0;

            std::vector<int> regionsList(
                connectorRegions[connX][connY].begin(),
                connectorRegions[connX][connY].end());

            int dest = regionsList[0];
            std::vector<int> sources(regionsList.begin()+1, regionsList.end());

            for(int i = 0; i <= currentRegion; i++) {
                if(std::find(sources.begin(), sources.end(), merged[i])
                   != sources.end()) {
                    merged[i] = dest;
                }
            }

            for(int source : sources) {
                openRegions.erase(source);
            }

            // Remove invalid connectors
            for(int x = 0; x < mapWidth; x++) {
                for(int y = 0; y < mapHeight; y++) {
                    if(connectorRegions[x][y].empty()) continue;

                    std::set<int> newRegions;
                    for(int region : connectorRegions[x][y]) {
                        newRegions.insert(merged[region]);
                    }

                    if(newRegions.size() == 1) {
                        if(std::uniform_real_distribution<>(0,1)(gen)
                           < CONNECTION_CHANCE) {
                            map[x][y] = 0;
                        }
                        connectorRegions[x][y].clear();
                    } else {
                        connectorRegions[x][y] = newRegions;
                    }
                }
            }
        }
    }

    void removeDeadEnds(int mapWidth, int mapHeight) {
        bool done = false;
        while(!done) {
            done = true;
            for(int y = 1; y < mapHeight-1; y++) {
                for(int x = 1; x < mapWidth-1; x++) {
                    if(map[x][y] == 0) {
                        int exits = 0;
                        const std::vector<std::pair<int,int>> directions = {
                            {0,-1}, {0,1}, {1,0}, {-1,0}
                        };

                        for(const auto& dir : directions) {
                            if(map[x+dir.first][y+dir.second] == 0) {
                                exits++;
                            }
                        }

                        if(exits <= 1) {
                            map[x][y] = 1;
                            done = false;
                        }
                    }
                }
            }
        }
    }

    void startNewRegion() {
        currentRegion++;
    }

    void carve(int x, int y) {
        map[x][y] = 0;
        regions[x][y] = currentRegion;
    }

    void createRoom(const Room& room) {
        for(int x = room.x1; x < room.x2; x++) {
            for(int y = room.y1; y < room.y2; y++) {
                carve(x, y);
            }
        }
    }
};

class MessyBSPGenerator : public DungeonGenerator {
private:
    const int MAX_LEAF_SIZE = 24;
    const int ROOM_MAX_SIZE = 15;
    const int ROOM_MIN_SIZE = 6;
    const bool SMOOTH_EDGES = true;
    const int SMOOTHING = 1;
    const int FILLING = 3;

    std::vector<Leaf*> leafs;

public:
    DungeonGenerator* clone() const override {
        return new MessyBSPGenerator(*this);
    }

    void generate() override {
        leafs.reserve(512);
        clearMap();
        leafs.clear();

        // Create root leaf
        auto rootLeaf = std::make_unique<Leaf>(0, 0, MAP_WIDTH, MAP_HEIGHT);
        leafs.push_back(rootLeaf.get());

        // Split leafs
        bool splitSuccessfully = true;
        while(splitSuccessfully) {
            splitSuccessfully = false;

            for(auto leaf : leafs) {
                if(!leaf->child1 && !leaf->child2) {
                    if(leaf->width > MAX_LEAF_SIZE ||
                       leaf->height > MAX_LEAF_SIZE ||
                       std::uniform_real_distribution<>(0,1)(gen) > 0.8) {
                        if(leaf->split()) {
                            leafs.push_back(leaf->child1.get());
                            leafs.push_back(leaf->child2.get());
                            splitSuccessfully = true;
                        }
                    }
                }
            }
        }

        // Create rooms
        for(auto& leaf : leafs) {
            if(!leaf->child1 && !leaf->child2) {
                createRooms(*leaf);
            }
        }

        // Connect rooms
        for(size_t i = 0; i < leafs.size()-1; i++) {
            if(leafs[i]->room && leafs[i+1]->room) {
                connectRooms(*leafs[i]->room, *leafs[i+1]->room);
            }
        }

        // Clean up
        if(SMOOTH_EDGES) {
            smoothMap();
        }
    }

private:
    void createRooms(Leaf& leaf) {
        int w = std::uniform_int_distribution<>(
            ROOM_MIN_SIZE,
            std::min(ROOM_MAX_SIZE, leaf.width-1))(gen);
        int h = std::uniform_int_distribution<>(
            ROOM_MIN_SIZE,
            std::min(ROOM_MAX_SIZE, leaf.height-1))(gen);
        int x = std::uniform_int_distribution<>(
            leaf.x, leaf.x + (leaf.width-1) - w)(gen);
        int y = std::uniform_int_distribution<>(
            leaf.y, leaf.y + (leaf.height-1) - h)(gen);

        leaf.room = std::make_unique<Room>(x, y, w, h);
        createRoom(*leaf.room);
    }

    void createRoom(const Room& room) {
        for(int x = room.x1 + 1; x < room.x2; x++) {
            for(int y = room.y1 + 1; y < room.y2; y++) {
                if(x > 0 && x < MAP_WIDTH-1 && y > 0 && y < MAP_HEIGHT-1) {
                    map[x][y] = 0;
                }
            }
        }
    }

    void connectRooms(const Room& room1, const Room& room2) {
        auto [x1, y1] = room1.center();
        auto [x2, y2] = room2.center();

        int currX = x1;
        int currY = y1;

        while(!((room2.x1 <= currX && currX <= room2.x2) &&
               (room2.y1 <= currY && currY <= room2.y2))) {
            float north = 1.0f;
            float south = 1.0f;
            float east = 1.0f;
            float west = 1.0f;

            // Weight based on target position
            if(currX < x2) east += 1.0f;
            else if(currX > x2) west += 1.0f;
            if(currY < y2) south += 1.0f;
            else if(currY > y2) north += 1.0f;

            // Normalize weights
            float total = north + south + east + west;
            north /= total;
            south /= total;
            east /= total;
            west /= total;

            // Choose direction
            float choice = std::uniform_real_distribution<>(0,1)(gen);
            int dx = 0, dy = 0;

            if(choice < north) {
                dy = -1;
            } else if(choice < north + south) {
                dy = 1;
            } else if(choice < north + south + east) {
                dx = 1;
            } else {
                dx = -1;
            }

            // Move and carve
            if(currX + dx > 0 && currX + dx < MAP_WIDTH-1 &&
               currY + dy > 0 && currY + dy < MAP_HEIGHT-1) {
                currX += dx;
                currY += dy;
                map[currX][currY] = 0;
            }
        }
    }

    void smoothMap() {
        for(int i = 0; i < 3; i++) {
            for(int x = 1; x < MAP_WIDTH-1; x++) {
                for(int y = 1; y < MAP_HEIGHT-1; y++) {
                    int walls = countAdjacentWalls(x, y);

                    if(map[x][y] == 1 && walls <= SMOOTHING) {
                        map[x][y] = 0;
                    } else if(map[x][y] == 0 && walls >= FILLING) {
                        map[x][y] = 1;
                    }
                }
            }
        }
    }

    int countAdjacentWalls(int x, int y) {
        int count = 0;
        if(map[x][y-1] == 1) count++;
        if(map[x][y+1] == 1) count++;
        if(map[x-1][y] == 1) count++;
        if(map[x+1][y] == 1) count++;
        return count;
    }
};


class ColorScheme {
public:
    static const std::vector<std::array<sf::Color, 4>> schemes;

    static const sf::Color& getWallForeground(int scheme) { return schemes[scheme][0]; }
    static const sf::Color& getWallBackground(int scheme) { return schemes[scheme][1]; }
    static const sf::Color& getGroundForeground(int scheme) { return schemes[scheme][2]; }
    static const sf::Color& getGroundBackground(int scheme) { return schemes[scheme][3]; }
};

const std::vector<std::array<sf::Color, 4>> ColorScheme::schemes = {
    // BLUE
    {
        sf::Color(100, 100, 100),  // wall_fore
        sf::Color(50, 50, 150),    // wall_back
        sf::Color(128, 128, 128),  // ground_fore
        sf::Color(10, 10, 10)      // ground_back
    },
    // MAUVE
    {
        sf::Color(50, 50, 50),     // wall_fore
        sf::Color(204, 153, 255),  // wall_back
        sf::Color(128, 128, 128),  // ground_fore
        sf::Color(10, 10, 10)      // ground_back
    },
    // GRAYSCALE
    {
        sf::Color::Black,          // wall_fore
        sf::Color(128, 128, 128),  // wall_back
        sf::Color::White,          // ground_fore
        sf::Color::Black           // ground_back
    },
    // TEXT ONLY
    {
        sf::Color::White,          // wall_fore
        sf::Color::Black,          // wall_back
        sf::Color::White,          // ground_fore
        sf::Color::Black           // ground_back
    }
};

class UserInterface {
private:
    sf::RenderWindow window;
    sf::Font font;
    int currentColorScheme = 0;

    std::unique_ptr<DungeonGenerator> currentGenerator;
    std::vector<std::unique_ptr<DungeonGenerator>> generators;

    struct HelpText {
        std::string key;
        std::string description;
        sf::Text text;
    };
    std::vector<HelpText> helpTexts;

    const float CELL_SIZE = 10.0f;

public:
    UserInterface() :
        window(sf::VideoMode(SCREEN_WIDTH, SCREEN_HEIGHT),
            "Roguelike Dungeon Generator") {

        window.setFramerateLimit(60);

        // if (!font.loadFromFile("arial.ttf")) {
        //     throw std::runtime_error("Could not load font");
        // }

        // Initialize generators
        generators.push_back(std::make_unique<TunnelingGenerator>());
        generators.push_back(std::make_unique<BSPGenerator>());
        generators.push_back(std::make_unique<DrunkardWalkGenerator>());
        generators.push_back(std::make_unique<CellularAutomataGenerator>());
        generators.push_back(std::make_unique<RoomAdditionGenerator>());
        generators.push_back(std::make_unique<CityWallsGenerator>());
        generators.push_back(std::make_unique<MazeWithRoomsGenerator>());
        generators.push_back(std::make_unique<MessyBSPGenerator>());

        currentGenerator = std::make_unique<TunnelingGenerator>();
        currentGenerator->generate();

        initializeHelpText();
    }

    void run() {
        while (window.isOpen()) {
            handleEvents();
            render();
        }
    }

private:
    void handleEvents() {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            else if (event.type == sf::Event::KeyPressed) {
                handleKeyPress(event.key.code);
            }
        }
    }

    void handleKeyPress(sf::Keyboard::Key key) {
        switch (key) {
            case sf::Keyboard::Escape:
                window.close();
                break;

            case sf::Keyboard::Space:
                if (currentGenerator) {
                    currentGenerator->generate();
                }
                break;

            case sf::Keyboard::Num0:
                currentColorScheme = (currentColorScheme + 1) % ColorScheme::schemes.size();
                break;

            case sf::Keyboard::Num1:
                switchGenerator(0);
                break;

            case sf::Keyboard::Num2:
                switchGenerator(1);
                break;

            case sf::Keyboard::Num3:
                switchGenerator(2);
                break;

            case sf::Keyboard::Num4:
                switchGenerator(3);
                break;

            case sf::Keyboard::Num5:
                switchGenerator(4);
                break;

            case sf::Keyboard::Num6:
                switchGenerator(5);
                break;

            case sf::Keyboard::Num7:
                switchGenerator(6);
                break;

            case sf::Keyboard::Num8:
                switchGenerator(7);
                break;
        }
    }

    void switchGenerator(size_t index) {
        if (index < generators.size()) {
            currentGenerator = std::unique_ptr<DungeonGenerator>(
                generators[index]->clone());
            currentGenerator->generate();
        }
    }

    void render() {
        window.clear(sf::Color::Black);

        if (currentGenerator) {
            renderDungeon();
        }

        renderHelpText();

        window.display();
    }

    void renderDungeon() {
        const auto& map = currentGenerator->getMap();

        sf::RectangleShape cell(sf::Vector2f(CELL_SIZE - 1, CELL_SIZE - 1));

        for (int y = 0; y < MAP_HEIGHT; ++y) {
            for (int x = 0; x < MAP_WIDTH; ++x) {
                cell.setPosition(x * CELL_SIZE, y * CELL_SIZE);

                if (map[x][y] == 1) {
                    cell.setFillColor(ColorScheme::getWallBackground(currentColorScheme));
                    cell.setOutlineColor(ColorScheme::getWallForeground(currentColorScheme));
                } else {
                    cell.setFillColor(ColorScheme::getGroundBackground(currentColorScheme));
                    cell.setOutlineColor(ColorScheme::getGroundForeground(currentColorScheme));
                }

                cell.setOutlineThickness(1.0f);
                window.draw(cell);
            }
        }
    }

    void initializeHelpText() {
        const std::vector<std::pair<std::string, std::string>> texts = {
            {"1", "Tunneling Algorithm"},
            {"2", "BSP Tree Algorithm"},
            {"3", "Random Walk Algorithm"},
            {"4", "Cellular Automata"},
            {"5", "Room Addition"},
            {"6", "City Buildings"},
            {"7", "Maze with Rooms"},
            {"8", "Messy BSP Tree"},
            {"0", "Change Color Scheme"},
            {"Space", "Remake Dungeon"}
        };

        float y = SCREEN_HEIGHT * CELL_SIZE - 120;
        for (const auto& [key, desc] : texts) {
            HelpText help{key, desc};
            help.text.setFont(font);
            help.text.setCharacterSize(14);
            help.text.setFillColor(sf::Color::White);
            help.text.setString(key + ": " + desc);
            help.text.setPosition(10, y);
            helpTexts.push_back(help);
            y += 20;
        }
    }

    void renderHelpText() {
        for (const auto& help : helpTexts) {
            window.draw(help.text);
        }
    }
};

int main() {
    try {
        UserInterface ui;
        ui.run();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}