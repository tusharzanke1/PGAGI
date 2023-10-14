import { v4 as uuidv4 } from 'uuid'

import About from '@l3-lib/ui-core/dist/icons/About'
import Home from '@l3-lib/ui-core/dist/icons/Home'

import API from '@l3-lib/ui-core/dist/icons/API'
import Doc from '@l3-lib/ui-core/dist/icons/Doc'
import Games from '@l3-lib/ui-core/dist/icons/Games'
import Teams from '@l3-lib/ui-core/dist/icons/Teams'
import Players from '@l3-lib/ui-core/dist/icons/Players'
import Contracts from '@l3-lib/ui-core/dist/icons/Contracts'
import Collection from '@l3-lib/ui-core/dist/icons/Collection'
import Value from '@l3-lib/ui-core/dist/icons/Value'
import Add from '@l3-lib/ui-core/dist/icons/Add'
import Sun from '@l3-lib/ui-core/dist/icons/Sun'
import HomeIconSvg from 'assets/svgComponents/HomeIconSvg'
import StarVector from 'assets/svgComponents/StarVector'
import { StyledValueIcon } from 'pages/Navigation/MainNavigation'
import styled from 'styled-components'

export const defaultData = (path_id?: any) => {
  return [
    {
      id: uuidv4(),
      name: 'Home',
      url: '/',
      option: 'link',
      group_name: 'go_to',
      icon: <StyledHomeIcon />,
    },

    {
      id: uuidv4(),
      name: 'Agents',
      url: '/agents',
      option: 'link',
      group_name: ['go_to'],
      icon: <StyledCollectionIcon />,
    },
    {
      id: uuidv4(),
      name: 'Datasources',
      url: '/datasources',
      option: 'link',
      group_name: ['go_to'],
      icon: (
        <StyledValueIcon>
          <Value />
        </StyledValueIcon>
      ),
    },
    {
      id: uuidv4(),
      name: 'Team Of Agents',
      url: '/team-of-agents',
      option: 'link',
      group_name: ['go_to'],
      icon: <StyledTeamsIcon />,
    },
    {
      id: uuidv4(),
      name: 'Toolkits',
      url: '/toolkits',
      option: 'link',
      group_name: ['go_to'],
      icon: <StyledAddIcon />,
    },
    // {
    //   id: uuidv4(),
    //   name: 'Teams',
    //   url: '/teams',
    //   option: 'link',
    //   group_name: ['go_to'],
    //   icon: <Teams />,
    // },

    {
      id: uuidv4(),
      name: 'Create agent',
      url: '/agents/create-agent',
      option: 'link',
      group_name: ['go_to'],
      icon: <StyledCollectionIcon />,
    },

    {
      id: uuidv4(),
      name: 'Add datasource',
      url: '/datasources/create-datasource',
      option: 'link',
      group_name: ['go_to'],
      icon: (
        <StyledValueIcon>
          <Value />
        </StyledValueIcon>
      ),
    },

    // {
    //   id: uuidv4(),
    //   name: 'General',
    //   url: '/chat',
    //   option: 'open-chat',
    //   group_name: 'chat',
    //   icon: <Home />,
    // },

    // {
    //   id: uuidv4(),
    //   name: 'Agents List',
    //   url: '/agents',
    //   option: 'show-agents',
    //   group_name: 'go_to',
    //   icon: <Players />,
    // },

    // {
    //   id: uuidv4(),
    //   name: 'Collections',
    //   url: '/collections',
    //   option: 'show-collections',
    //   group_name: 'go_to',
    //   icon: <Contracts />,
    // },

    // {
    //   id: uuidv4(),
    //   name: 'Change Password',
    //   url: '/change-password',
    //   option: 'modal',
    //   group_name: 'go_to',
    //   icon: <Players />,
    // },
    {
      id: uuidv4(),
      name: 'Profile',
      url: '/account',
      option: 'modal',
      group_name: 'go_to',
      icon: <StyledAboutIcon />,
    },
    {
      id: uuidv4(),
      name: 'Logout',
      url: 'create',
      option: 'modal',
      // group_name: 'go_to',
      icon: <StyledAboutIcon />,
    },
    {
      id: uuidv4(),
      name: 'Set-blue-theme',
      option: 'theme',
      // group_name: '',
      icon: <StyledSunIcon />,
    },
    {
      id: uuidv4(),
      name: 'Set light theme',
      option: 'theme',
      // group_name: '',
      icon: <StyledSunIcon />,
    },
  ]
}

const StyledSunIcon = styled(Sun)`
  path {
    fill: ${({ theme }) => theme.body.iconColor};
  }
`

const StyledAboutIcon = styled(About)`
  path {
    fill: ${({ theme }) => theme.body.iconColor};
  }
`

const StyledHomeIcon = styled(HomeIconSvg)`
  path {
    fill: ${({ theme }) => theme.body.iconColor};
  }
`

const StyledCollectionIcon = styled(Collection)`
  path {
    fill: ${({ theme }) => theme.body.iconColor};
  }
`
const StyledAddIcon = styled(Add)`
  path {
    fill: ${({ theme }) => theme.body.iconColor};
  }
`

const StyledTeamsIcon = styled(Teams)`
  path {
    fill: ${({ theme }) => theme.body.iconColor};
  }
`

const StyledIcon = styled(Value)`
  path {
    fill: ${({ theme }) => theme.body.iconColor};
  }
`
