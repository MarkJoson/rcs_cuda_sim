#include <SFML/Graphics.hpp>
#include <iostream>
#include "map_generator.h"

using namespace map_gen;


constexpr int SCREEN_WIDTH = 800;
constexpr int SCREEN_HEIGHT = 600;

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

        for (int y = 0; y < currentGenerator->getWidth(); ++y) {
            for (int x = 0; x < currentGenerator->getHeight(); ++x) {
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