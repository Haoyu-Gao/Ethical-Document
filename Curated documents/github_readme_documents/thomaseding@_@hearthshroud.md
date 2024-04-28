### About

Hearthshroud is one of two things:
 * A Hearthstone game engine library.
 * A simple Hearthstone executable with a console UI.

--------------------

### Library

Library uses a monadic API which drives the game engine for any `HearthMonad`:
```haskell
-- MonadPrompt @ https://hackage.haskell.org/package/MonadPrompt
type HearthMonad m = MonadPrompt HearthPrompt m
Hearth.Engine.runHearth :: (HearthMonad m) => Pair PlayerData -> m GameResult
```

##### Design Choices
 * Model cards (and abilities, effects, etc.) as a pure data AST. This is the card DSL.
 * Model enforces game constraints at the type level.

##### Design Implications 
 * Engine interprets the cards. (As opposed to the cards directly manipulating the environment.)
 * AI directly interprets the same cards as well.

--------------------

### Executable

Sample game client is `Hearth.Client.Console` and can be seen in action by:
```haskell
Hearth.Client.Console.main :: IO ()
```
![ui-game-1](https://cloud.githubusercontent.com/assets/6971794/11055545/16f84d62-872d-11e5-8745-7fcf35add15d.png)
![ui-game-2](https://cloud.githubusercontent.com/assets/6971794/11055306/e807bed6-872a-11e5-9d54-7ffd9a3c7d82.png)
![ui-show-card](https://cloud.githubusercontent.com/assets/6971794/9697842/382720c0-5353-11e5-925b-bbf4665854bf.png)
![ui-game-help](https://cloud.githubusercontent.com/assets/6971794/9697844/84ef6a20-5353-11e5-9e3e-21369cd81479.png)
![ui-program-help](https://cloud.githubusercontent.com/assets/6971794/9697852/df37c482-5353-11e5-862a-4b349f239c11.png)

--------------------

### Installation

#### Steps
 * Clone repository
 * Use `cabal` tool to build/install

For those unfamiliar with `cabal`, search Google for "haskell cabal how to use" (or whatever).

#### Contact Me
 * Probably the best way to ask me a question or tell me a comment is to just create an issue directly for the project (even if it is not a real issue).
 * I am sometimes on `#hearthsim` IRC.

--------------------

### Related Hearthstone Projects

 * http://hearthsim.info/
